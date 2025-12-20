#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include "cifar10_dataset.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 32
#define GRID(n) ((n + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Optimized GEMM tile sizes
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 4
#define THREAD_N 4

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err); \
    } \
} while(0)

// ============== MEMORY POOL ==============
class MemoryPool {
    std::vector<std::pair<float*, size_t>> buffers;
    size_t total = 0;
public:
    float* alloc(size_t bytes) {
        float* p; cudaMalloc(&p, bytes);
        buffers.push_back({p, bytes});
        total += bytes;
        return p;
    }
    size_t get_total() const { return total; }
    ~MemoryPool() { for (auto& b : buffers) cudaFree(b.first); }
};

// ============== FUSED GEMM + BIAS + RELU KERNELS ==============

// C[M,N] = ReLU(A[M,K] * B^T[N,K] + bias[N])
__global__ void gemm_nt_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int K, int N, bool relu)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_col = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && b_col < K) ? B[col * K + b_col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float val = sum + bias[col];
        C[row * N + col] = relu ? fmaxf(val, 0.0f) : val;
    }
}

// Optimized version with register blocking
__global__ void gemm_nt_bias_relu_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int K, int N, bool relu)
{
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    float acc[THREAD_M][THREAD_N] = {0.0f};
    
    int row_base = by * TILE_M;
    int col_base = bx * TILE_N;
    
    int threads_per_block = blockDim.x * blockDim.y;
    
    for (int k = 0; k < K; k += TILE_K) {
        for (int i = tid; i < TILE_K * TILE_M; i += threads_per_block) {
            int ki = i / TILE_M;
            int mi = i % TILE_M;
            int global_row = row_base + mi;
            int global_k = k + ki;
            As[ki][mi] = (global_row < M && global_k < K) ? A[global_row * K + global_k] : 0.0f;
        }
        
        for (int i = tid; i < TILE_K * TILE_N; i += threads_per_block) {
            int ki = i / TILE_N;
            int ni = i % TILE_N;
            int global_col = col_base + ni;
            int global_k = k + ki;
            Bs[ki][ni] = (global_col < N && global_k < K) ? B[global_col * K + global_k] : 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int ki = 0; ki < TILE_K; ++ki) {
            float a_reg[THREAD_M], b_reg[THREAD_N];
            
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_reg[m] = As[ki][ty * THREAD_M + m];
            }
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                b_reg[n] = Bs[ki][tx * THREAD_N + n];
            }
            
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                #pragma unroll
                for (int n = 0; n < THREAD_N; ++n) {
                    acc[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        int global_row = row_base + ty * THREAD_M + m;
        #pragma unroll
        for (int n = 0; n < THREAD_N; ++n) {
            int global_col = col_base + tx * THREAD_N + n;
            if (global_row < M && global_col < N) {
                float val = acc[m][n] + bias[global_col];
                C[global_row * N + global_col] = relu ? fmaxf(val, 0.0f) : val;
            }
        }
    }
}

// Wrapper with automatic kernel selection
void gemm_nt_bias_relu(const float* A, const float* B, const float* bias, 
                        float* C, int M, int K, int N, bool relu, cudaStream_t stream) {
    if (M >= 64 && N >= 64 && K >= 16) {
        dim3 block(TILE_N / THREAD_N, TILE_M / THREAD_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        gemm_nt_bias_relu_optimized_kernel<<<grid, block, 0, stream>>>(A, B, bias, C, M, K, N, relu);
    } else {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        gemm_nt_bias_relu_kernel<<<grid, block, 0, stream>>>(A, B, bias, C, M, K, N, relu);
    }
}

// ============== IM2COL KERNEL ==============
__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ col,
    int B, int H, int W, int C,
    int K, int P, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H_out * W_out * C * K * K;
    if (idx >= total) return;
    
    int kk = idx % (K * K);
    int tmp = idx / (K * K);
    int c = tmp % C;
    tmp /= C;
    int ow = tmp % W_out;
    tmp /= W_out;
    int oh = tmp % H_out;
    int b = tmp / H_out;
    
    int kh = kk / K;
    int kw = kk % K;
    int ih = oh - P + kh;
    int iw = ow - P + kw;
    
    int col_row = b * (H_out * W_out) + oh * W_out + ow;
    int col_col = c * K * K + kh * K + kw;
    int col_width = C * K * K;
    
    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = input[b * (H * W * C) + ih * (W * C) + iw * C + c];
    }
    col[col_row * col_width + col_col] = val;
}

// ============== MAXPOOL KERNEL ==============
__global__ void maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int H_in, int W_in, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2, W_out = W_in / 2;
    int total = B * H_out * W_out * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int tmp = idx / C;
    int wo = tmp % W_out;
    tmp /= W_out;
    int ho = tmp % H_out;
    int b = tmp / H_out;
    
    int hi = ho * 2, wi = wo * 2;
    float max_val = -1e10f;
    
    #pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
        #pragma unroll
        for (int dw = 0; dw < 2; ++dw) {
            int in_idx = b * (H_in * W_in * C) + (hi + dh) * (W_in * C) + (wi + dw) * C + c;
            float v = input[in_idx];
            if (v > max_val) { max_val = v; }
        }
    }
    output[idx] = max_val;
}

// ============== UTILITY FUNCTIONS ==============

std::vector<float> load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(1);
    }

    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

    std::vector<float> weights(size);
    file.read(reinterpret_cast<char*>(weights.data()), size * sizeof(float));

    file.close();
    return weights;
}

void create_directory_if_not_exists(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
}

void save_features(const std::string& filename,
                   const std::vector<std::vector<float>>& features,
                   const std::vector<int>& labels) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << " for writing" << std::endl;
        std::cerr << "Make sure the directory exists or run: mkdir features" << std::endl;
        exit(1);
    }

    uint32_t num_samples = features.size();
    uint32_t feature_dim = features[0].size();
    file.write(reinterpret_cast<const char*>(&num_samples), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&feature_dim), sizeof(uint32_t));

    for (const auto& feature : features) {
        file.write(reinterpret_cast<const char*>(feature.data()), feature_dim * sizeof(float));
    }

    file.write(reinterpret_cast<const char*>(labels.data()), num_samples * sizeof(int));

    file.close();
    std::cout << "Saved " << num_samples << " samples to " << filename << std::endl;
}

// ============== MAIN ==============

int main() {
    std::cout << "=== Feature Extraction (Optimized GPU) ===\n\n";
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n\n";
    
    // Load weights - matching filenames from train_gpu_optimize.cu
    std::cout << "Loading trained weights...\n";
    
    auto h_w1 = load_weights("../weights/enc_w1.bin");
    auto h_b1 = load_weights("../weights/enc_b1.bin");
    auto h_w2 = load_weights("../weights/enc_w2.bin");
    auto h_b2 = load_weights("../weights/enc_b2.bin");
        // Print an example from h_w1 (first 10 weights)
    std::cout << "\nExample weights from enc_w1.bin:\n";
    for (size_t i = 0; i < std::min((size_t)10, h_w1.size()); ++i) {
        std::cout << std::fixed << std::setprecision(5) << h_w1[i] << " ";
    }
    std::cout << "... (total " << h_w1.size() << " values)\n";
    std::cout << "  w1: " << h_w1.size() << " (" << 256 << "x" << 3*9 << ")\n";
    std::cout << "  b1: " << h_b1.size() << "\n";
    std::cout << "  w2: " << h_w2.size() << " (" << 128 << "x" << 256*9 << ")\n";
    std::cout << "  b2: " << h_b2.size() << "\n\n";
    
    // Load dataset
    std::cout << "Loading CIFAR-10 dataset...\n";
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin");
    dataset.load_data();
    
    const int NUM_TRAIN = dataset.get_num_train();
    const int NUM_TEST = dataset.get_num_test();
    const int FEATURE_DIM = 8 * 8 * 128;
    const int IMG_SIZE = 32 * 32 * 3;
    const int B = 64;
    
    std::cout << "  Train: " << NUM_TRAIN << "\n";
    std::cout << "  Test: " << NUM_TEST << "\n\n";
    
    // Memory pool
    MemoryPool pool;
    
    // Device memory for weights
    float *d_w1 = pool.alloc(h_w1.size() * 4);
    float *d_b1 = pool.alloc(h_b1.size() * 4);
    float *d_w2 = pool.alloc(h_w2.size() * 4);
    float *d_b2 = pool.alloc(h_b2.size() * 4);
    
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * 4, cudaMemcpyHostToDevice);
    
    // Intermediate buffers
    int s_in = B * 32 * 32 * 3;
    int s_l1 = B * 32 * 32 * 256;
    int s_p1 = B * 16 * 16 * 256;
    int s_l2 = B * 16 * 16 * 128;
    int s_p2 = B * 8 * 8 * 128;
    
    int col1_size = B * 32 * 32 * (3 * 9);
    int col2_size = B * 16 * 16 * (256 * 9);
    
    float *d_input = pool.alloc(s_in * 4);
    float *d_col1 = pool.alloc(col1_size * 4);
    float *d_l1 = pool.alloc(s_l1 * 4);
    float *d_p1 = pool.alloc(s_p1 * 4);
    float *d_col2 = pool.alloc(col2_size * 4);
    float *d_l2 = pool.alloc(s_l2 * 4);
    float *d_p2 = pool.alloc(s_p2 * 4);
    
    std::cout << "GPU Memory: " << pool.get_total() / (1024 * 1024) << " MB\n\n";
    
    // Pinned host memory
    float* h_pinned_input;
    float* h_pinned_features;
    cudaMallocHost(&h_pinned_input, s_in * 4);
    cudaMallocHost(&h_pinned_features, s_p2 * 4);
    
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Storage for features
    std::vector<std::vector<float>> X_train, X_test;
    std::vector<int> y_train, y_test;
    
    float* train_images = dataset.get_train_images_ptr();
    unsigned char* train_labels = dataset.get_train_labels_ptr();
    float* test_images = dataset.get_test_images_ptr();
    unsigned char* test_labels = dataset.get_test_labels_ptr();
    
    // ========== EXTRACT TRAIN FEATURES ==========
    std::cout << "Extracting train features...\n";
    int num_train_batches = (NUM_TRAIN + B - 1) / B;
    
    for (int batch = 0; batch < num_train_batches; ++batch) {
        int start_idx = batch * B;
        int current_B = std::min(B, NUM_TRAIN - start_idx);
        
        // Copy batch to GPU
        memcpy(h_pinned_input, train_images + start_idx * IMG_SIZE, current_B * IMG_SIZE * 4);
        cudaMemcpyAsync(d_input, h_pinned_input, current_B * IMG_SIZE * 4, cudaMemcpyHostToDevice, stream);
        
        // Layer 1: im2col -> GEMM+bias+ReLU -> MaxPool
        int actual_col1 = current_B * 32 * 32 * (3 * 9);
        im2col_kernel<<<GRID(actual_col1), BLOCK_SIZE, 0, stream>>>(
            d_input, d_col1, current_B, 32, 32, 3, 3, 1, 32, 32);
        
        gemm_nt_bias_relu(d_col1, d_w1, d_b1, d_l1, 
                          current_B * 32 * 32, 3 * 9, 256, true, stream);
        
        int actual_p1 = current_B * 16 * 16 * 256;
        maxpool_kernel<<<GRID(actual_p1), BLOCK_SIZE, 0, stream>>>(
            d_l1, d_p1, current_B, 32, 32, 256);
        
        // Layer 2: im2col -> GEMM+bias+ReLU -> MaxPool
        int actual_col2 = current_B * 16 * 16 * (256 * 9);
        im2col_kernel<<<GRID(actual_col2), BLOCK_SIZE, 0, stream>>>(
            d_p1, d_col2, current_B, 16, 16, 256, 3, 1, 16, 16);
        
        gemm_nt_bias_relu(d_col2, d_w2, d_b2, d_l2, 
                          current_B * 16 * 16, 256 * 9, 128, true, stream);
        
        int actual_p2 = current_B * 8 * 8 * 128;
        maxpool_kernel<<<GRID(actual_p2), BLOCK_SIZE, 0, stream>>>(
            d_l2, d_p2, current_B, 16, 16, 128);
        
        // Copy features back
        cudaMemcpyAsync(h_pinned_features, d_p2, current_B * FEATURE_DIM * 4, 
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Store features
        for (int j = 0; j < current_B; ++j) {
            std::vector<float> feat(h_pinned_features + j * FEATURE_DIM,
                                    h_pinned_features + (j + 1) * FEATURE_DIM);
            X_train.push_back(feat);
            y_train.push_back(static_cast<int>(train_labels[start_idx + j]));
        }
        
        if ((batch + 1) % 100 == 0 || batch == num_train_batches - 1) {
            std::cout << "  " << std::min((batch + 1) * B, NUM_TRAIN) << "/" << NUM_TRAIN << "\n";
        }
    }
    
    // ========== EXTRACT TEST FEATURES ==========
    std::cout << "\nExtracting test features...\n";
    int num_test_batches = (NUM_TEST + B - 1) / B;
    
    for (int batch = 0; batch < num_test_batches; ++batch) {
        int start_idx = batch * B;
        int current_B = std::min(B, NUM_TEST - start_idx);
        
        memcpy(h_pinned_input, test_images + start_idx * IMG_SIZE, current_B * IMG_SIZE * 4);
        cudaMemcpyAsync(d_input, h_pinned_input, current_B * IMG_SIZE * 4, cudaMemcpyHostToDevice, stream);
        
        // Layer 1
        int actual_col1 = current_B * 32 * 32 * (3 * 9);
        im2col_kernel<<<GRID(actual_col1), BLOCK_SIZE, 0, stream>>>(
            d_input, d_col1, current_B, 32, 32, 3, 3, 1, 32, 32);
        
        gemm_nt_bias_relu(d_col1, d_w1, d_b1, d_l1, 
                          current_B * 32 * 32, 3 * 9, 256, true, stream);
        
        int actual_p1 = current_B * 16 * 16 * 256;
        maxpool_kernel<<<GRID(actual_p1), BLOCK_SIZE, 0, stream>>>(
            d_l1, d_p1, current_B, 32, 32, 256);
        
        // Layer 2
        int actual_col2 = current_B * 16 * 16 * (256 * 9);
        im2col_kernel<<<GRID(actual_col2), BLOCK_SIZE, 0, stream>>>(
            d_p1, d_col2, current_B, 16, 16, 256, 3, 1, 16, 16);
        
        gemm_nt_bias_relu(d_col2, d_w2, d_b2, d_l2, 
                          current_B * 16 * 16, 256 * 9, 128, true, stream);
        
        int actual_p2 = current_B * 8 * 8 * 128;
        maxpool_kernel<<<GRID(actual_p2), BLOCK_SIZE, 0, stream>>>(
            d_l2, d_p2, current_B, 16, 16, 128);
        
        cudaMemcpyAsync(h_pinned_features, d_p2, current_B * FEATURE_DIM * 4, 
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        for (int j = 0; j < current_B; ++j) {
            std::vector<float> feat(h_pinned_features + j * FEATURE_DIM,
                                    h_pinned_features + (j + 1) * FEATURE_DIM);
            X_test.push_back(feat);
            y_test.push_back(static_cast<int>(test_labels[start_idx + j]));
        }
        
        if ((batch + 1) % 50 == 0 || batch == num_test_batches - 1) {
            std::cout << "  " << std::min((batch + 1) * B, NUM_TEST) << "/" << NUM_TEST << "\n";
        }
    }
    
    std::cout << "\nFeature extraction complete!\n";
    std::cout << "  Train: (" << X_train.size() << ", " << FEATURE_DIM << ")\n";
    std::cout << "  Test: (" << X_test.size() << ", " << FEATURE_DIM << ")\n\n";
    std::cout << "\nExample feature (first train sample):\n";
    if (!X_train.empty()) {
        for (size_t i = 0; i < std::min((size_t)20, X_train[0].size()); ++i) { // Print first 20 elements
            std::cout << std::fixed << std::setprecision(5) << X_train[0][i] << " ";
        }
        std::cout << "... (total " << X_train[0].size() << " values)\n";
    }
    // Save features
    std::cout << "Saving features...\n";
    create_directory_if_not_exists("../features");
    save_features("../features/train_features.bin", X_train, y_train);
    save_features("../features/test_features.bin", X_test, y_test);
    
    // Cleanup
    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_features);
    cudaStreamDestroy(stream);
    
    std::cout << "\nDone!\n";
    return 0;
}