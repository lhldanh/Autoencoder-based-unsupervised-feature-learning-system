%%writefile src/train_gpu_optimize.cu
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
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

// ============== FUSED GEMM + BIAS + RELU KERNELS (FORWARD) ==============

// C[M,N] = ReLU(A[M,K] * B^T[N,K] + bias[N])  (B is transposed, fused bias+relu)
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

// Optimized version with register blocking - fused bias + relu
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

// Wrapper with automatic kernel selection for fused GEMM+bias+relu
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

// ============== STANDARD GEMM KERNELS ==============

// C[M,N] = A[M,K] * B[K,N]
__global__ void gemm_nn_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// C[M,N] = A^T[K,M] * B[K,N]  (A is transposed)
__global__ void gemm_tn_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int a_row = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (a_row < K && row < M) ? A[a_row * M + row] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[k][threadIdx.y] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void gemm_nn(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_nn_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

void gemm_tn(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tn_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
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

// ============== COL2IM KERNEL ==============
__global__ void col2im_kernel(
    const float* __restrict__ col,
    float* __restrict__ input_grad,
    int B, int H, int W, int C,
    int K, int P, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * C;
    if (idx >= total) return;
    
    int ic = idx % C;
    int tmp = idx / C;
    int iw = tmp % W;
    tmp /= W;
    int ih = tmp % H;
    int b = tmp / H;
    
    float sum = 0.0f;
    int col_width = C * K * K;
    
    #pragma unroll
    for (int kh = 0; kh < K; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            int oh = ih + P - kh;
            int ow = iw + P - kw;
            
            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                int col_row = b * (H_out * W_out) + oh * W_out + ow;
                int col_col = ic * K * K + kh * K + kw;
                sum += col[col_row * col_width + col_col];
            }
        }
    }
    input_grad[idx] = sum;
}

// ============== FUSED MAXPOOL ==============
__global__ void maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ indices,
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
    int max_idx = 0;
    
    #pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
        #pragma unroll
        for (int dw = 0; dw < 2; ++dw) {
            int in_idx = b * (H_in * W_in * C) + (hi + dh) * (W_in * C) + (wi + dw) * C + c;
            float v = input[in_idx];
            if (v > max_val) { max_val = v; max_idx = in_idx; }
        }
    }
    output[idx] = max_val;
    indices[idx] = max_idx;
}

// ============== UPSAMPLE ==============
__global__ void upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int H_in, int W_in, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in * 2, W_out = W_in * 2;
    int total = B * H_out * W_out * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int tmp = idx / C;
    int wo = tmp % W_out;
    tmp /= W_out;
    int ho = tmp % H_out;
    int b = tmp / H_out;
    
    output[idx] = input[b * (H_in * W_in * C) + (ho / 2) * (W_in * C) + (wo / 2) * C + c];
}

// ============== FUSED BACKWARD KERNELS ==============

// Fused MSE loss + backward in single kernel
__global__ void mse_loss_backward_fused_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ grad,
    float* __restrict__ partial_loss,
    int size)
{
    __shared__ float s[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float local_sum = 0.0f;
    float inv_size = 2.0f / size;
    
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float d = pred[i] - target[i];
        grad[i] = d * inv_size;
        local_sum += d * d;
    }
    
    s[tid] = local_sum;
    __syncthreads();
    
    for (int i = 128; i > 0; i >>= 1) {
        if (tid < i) s[tid] += s[tid + i];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial_loss, s[0]);
}

// ============== FUSED UPSAMPLE + RELU BACKWARD ==============
__global__ void fused_upsample_relu_backward_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ fwd,
    float* __restrict__ d_in,
    int B, int H_in, int W_in, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H_in * W_in * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int tmp = idx / C;
    int wi = tmp % W_in;
    tmp /= W_in;
    int hi = tmp % H_in;
    int b = tmp / H_in;
    
    int H_out = H_in * 2, W_out = W_in * 2;
    int ho = hi * 2, wo = wi * 2;
    
    // Upsample backward: sum 2x2 region
    float sum = d_out[b * (H_out * W_out * C) + ho * (W_out * C) + wo * C + c]
              + d_out[b * (H_out * W_out * C) + ho * (W_out * C) + (wo + 1) * C + c]
              + d_out[b * (H_out * W_out * C) + (ho + 1) * (W_out * C) + wo * C + c]
              + d_out[b * (H_out * W_out * C) + (ho + 1) * (W_out * C) + (wo + 1) * C + c];
    
    // Fused ReLU backward
    d_in[idx] = (fwd[idx] > 0.0f) ? sum : 0.0f;
}

void fused_upsample_relu_backward(const float* d_out, const float* fwd, float* d_in,
                                   int B, int H_in, int W_in, int C, cudaStream_t stream) {
    int total = B * H_in * W_in * C;
    fused_upsample_relu_backward_kernel<<<GRID(total), BLOCK_SIZE, 0, stream>>>(
        d_out, fwd, d_in, B, H_in, W_in, C);
}


// ============== VECTORIZED UTILITY KERNELS ==============

__global__ void fill_zeros_vectorized_kernel(float* data, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        *reinterpret_cast<float4*>(&data[idx]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) data[i] = 0.0f;
    }
}

__global__ void sgd_vectorized_kernel(float* w, const float* g, int size, float lr) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 w4 = *reinterpret_cast<float4*>(&w[idx]);
        float4 g4 = *reinterpret_cast<const float4*>(&g[idx]);
        w4.x -= lr * g4.x;
        w4.y -= lr * g4.y;
        w4.z -= lr * g4.z;
        w4.w -= lr * g4.w;
        *reinterpret_cast<float4*>(&w[idx]) = w4;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            w[i] -= lr * g[i];
        }
    }
}

__global__ void sgd_kernel(float* w, const float* g, int size, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) w[i] -= lr * g[i];
}

// ============== FUSED MAXPOOL + RELU BACKWARD ==============
__global__ void fused_maxpool_relu_backward_kernel(
    const float* __restrict__ d_out,
    const int* __restrict__ indices,
    const float* __restrict__ fwd,
    float* __restrict__ d_in,
    int pool_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pool_size) return;
    
    int target_idx = indices[i];
    float grad = d_out[i];
    
    // Fused ReLU backward: only propagate if forward was > 0
    if (fwd[target_idx] > 0.0f) {
        atomicAdd(&d_in[target_idx], grad);
    }
}

void fused_maxpool_relu_backward(const float* d_out, const int* indices, const float* fwd,
                                  float* d_in, int pool_size, int input_size, cudaStream_t stream) {
    // First zero d_in
    fill_zeros_vectorized_kernel<<<GRID(input_size / 4), BLOCK_SIZE, 0, stream>>>(d_in, input_size);
    // Then scatter with fused relu
    fused_maxpool_relu_backward_kernel<<<GRID(pool_size), BLOCK_SIZE, 0, stream>>>(
        d_out, indices, fwd, d_in, pool_size);
}

// ============== FUSED GEMM_NN + RELU BACKWARD (for input gradient) ==============
// Computes: d_col = ReLU_backward(d_out, fwd) * W
__global__ void gemm_nn_relu_backward_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ fwd,
    const float* __restrict__ W,
    float* __restrict__ d_col,
    int M, int K, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        // Load A with ReLU backward mask applied
        float a_val = 0.0f;
        if (row < M && a_col < K) {
            float fwd_val = fwd[row * K + a_col];
            float grad_val = d_out[row * K + a_col];
            a_val = (fwd_val > 0.0f) ? grad_val : 0.0f;
        }
        As[threadIdx.y][threadIdx.x] = a_val;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? W[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        d_col[row * N + col] = sum;
    }
}

void gemm_nn_relu_backward(const float* d_out, const float* fwd, const float* W,
                            float* d_col, int M, int K, int N, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_nn_relu_backward_kernel<<<grid, block, 0, stream>>>(d_out, fwd, W, d_col, M, K, N);
}

// ============== FUSED GEMM_TN + RELU BACKWARD (for weight gradient) ==============
// Computes: dW = (ReLU_backward(d_out, fwd))^T * col
__global__ void gemm_tn_relu_backward_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ fwd,
    const float* __restrict__ col,
    float* __restrict__ dW,
    int M, int K, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int a_row = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        // Load A^T with ReLU backward mask
        float a_val = 0.0f;
        if (a_row < K && row < M) {
            float fwd_val = fwd[a_row * M + row];
            float grad_val = d_out[a_row * M + row];
            a_val = (fwd_val > 0.0f) ? grad_val : 0.0f;
        }
        As[threadIdx.y][threadIdx.x] = a_val;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col_idx < N) ? col[b_row * N + col_idx] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[k][threadIdx.y] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col_idx < N) {
        dW[row * N + col_idx] = sum;
    }
}

void gemm_tn_relu_backward(const float* d_out, const float* fwd, const float* col,
                            float* dW, int M, int K, int N, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tn_relu_backward_kernel<<<grid, block, 0, stream>>>(d_out, fwd, col, dW, M, K, N);
}

// ============== FUSED BIAS BACKWARD + RELU ==============
__global__ void bias_backward_relu_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ fwd,
    float* __restrict__ d_bias,
    int B_HW, int C)
{
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int oc = blockIdx.x;
    if (oc >= C) return;
    
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    for (int i = tid; i < B_HW; i += BLOCK_SIZE) {
        int idx = i * C + oc;
        float fwd_val = fwd[idx];
        float grad_val = d_out[idx];
        local_sum += (fwd_val > 0.0f) ? grad_val : 0.0f;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128]; __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64]; __syncthreads();
    
    if (tid < 32) {
        volatile float* vs = shared_sum;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }
    
    if (tid == 0) d_bias[oc] = shared_sum[0];
}

// ============== NON-FUSED BACKWARD KERNELS (for layers without ReLU) ==============

__global__ void relu_backward_kernel(const float* d_out, const float* fwd, float* d_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_in[i] = (fwd[i] > 0.0f) ? d_out[i] : 0.0f;
}

__global__ void maxpool_backward_kernel(const float* d_out, const int* idx, float* d_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) atomicAdd(&d_in[idx[i]], d_out[i]);
}

__global__ void upsample_backward_kernel(const float* d_out, float* d_in, int B, int H_in, int W_in, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H_in * W_in * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int tmp = idx / C;
    int wi = tmp % W_in;
    tmp /= W_in;
    int hi = tmp % H_in;
    int b = tmp / H_in;
    
    int H_out = H_in * 2, W_out = W_in * 2;
    int ho = hi * 2, wo = wi * 2;
    
    float sum = d_out[b * (H_out * W_out * C) + ho * (W_out * C) + wo * C + c]
              + d_out[b * (H_out * W_out * C) + ho * (W_out * C) + (wo + 1) * C + c]
              + d_out[b * (H_out * W_out * C) + (ho + 1) * (W_out * C) + wo * C + c]
              + d_out[b * (H_out * W_out * C) + (ho + 1) * (W_out * C) + (wo + 1) * C + c];
    d_in[idx] = sum;
}

__global__ void bias_backward_kernel(const float* d_out, float* d_bias, int B_HW, int C) {
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int oc = blockIdx.x;
    if (oc >= C) return;
    
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    for (int i = tid; i < B_HW; i += BLOCK_SIZE) {
        local_sum += d_out[i * C + oc];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    if (tid < 128) shared_sum[tid] += shared_sum[tid + 128]; __syncthreads();
    if (tid < 64) shared_sum[tid] += shared_sum[tid + 64]; __syncthreads();
    
    if (tid < 32) {
        volatile float* vs = shared_sum;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }
    
    if (tid == 0) d_bias[oc] = shared_sum[0];
}


// ============== HELPERS ==============
void init_random(std::vector<float>& v, int fan_in, int fan_out) {
    std::mt19937 gen(42);
    float lim = sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> d(-lim, lim);
    for (auto& x : v) x = d(gen);
}

void save_weights(const std::string& f, const std::vector<float>& d) {
    std::ofstream file(f, std::ios::binary);
    uint32_t sz = d.size();
    file.write((char*)&sz, 4);
    file.write((char*)d.data(), d.size() * 4);
}

// ...existing code (all kernels and helper functions)...

int main() {
    const int B = 64, EPOCHS = 10;
    const float LR = 0.001f;
    
    std::cout << "=== CUDA Autoencoder (Fused Backward Kernels) ===\n\n";
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";
    
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin");
    dataset.load_data();
    if (dataset.get_num_train() == 0) { std::cerr << "No data!\n"; return 1; }
    std::cout << "Images: " << dataset.get_num_train() << "\n\n";
    
    MemoryPool pool;
    
    // Layer dimensions
    int s_in = B * 32 * 32 * 3;
    int s_l1 = B * 32 * 32 * 256, s_p1 = B * 16 * 16 * 256;
    int s_l2 = B * 16 * 16 * 128, s_p2 = B * 8 * 8 * 128;
    int s_l3 = B * 8 * 8 * 128,   s_u3 = B * 16 * 16 * 128;
    int s_l4 = B * 16 * 16 * 256, s_u4 = B * 32 * 32 * 256;
    
    // Col buffer sizes
    int col1_size = B * 32 * 32 * (3 * 9);
    int col2_size = B * 16 * 16 * (256 * 9);
    int col3_size = B * 8 * 8 * (128 * 9);
    int col4_size = B * 16 * 16 * (128 * 9);
    int col5_size = B * 32 * 32 * (256 * 9);
    
    // Weights
    std::vector<float> h_w1(256 * 3 * 9), h_b1(256, 0);
    std::vector<float> h_w2(128 * 256 * 9), h_b2(128, 0);
    std::vector<float> h_w3(128 * 128 * 9), h_b3(128, 0);
    std::vector<float> h_w4(256 * 128 * 9), h_b4(256, 0);
    std::vector<float> h_w5(3 * 256 * 9), h_b5(3, 0);
    
    init_random(h_w1, 27, 256); init_random(h_w2, 2304, 128);
    init_random(h_w3, 1152, 128); init_random(h_w4, 1152, 256);
    init_random(h_w5, 2304, 3);
    
    // Device memory - weights
    float *d_w1, *d_b1, *d_dw1, *d_db1;
    float *d_w2, *d_b2, *d_dw2, *d_db2;
    float *d_w3, *d_b3, *d_dw3, *d_db3;
    float *d_w4, *d_b4, *d_dw4, *d_db4;
    float *d_w5, *d_b5, *d_dw5, *d_db5;
    
    d_w1 = pool.alloc(h_w1.size() * 4); d_b1 = pool.alloc(256 * 4);
    d_dw1 = pool.alloc(h_w1.size() * 4); d_db1 = pool.alloc(256 * 4);
    d_w2 = pool.alloc(h_w2.size() * 4); d_b2 = pool.alloc(128 * 4);
    d_dw2 = pool.alloc(h_w2.size() * 4); d_db2 = pool.alloc(128 * 4);
    d_w3 = pool.alloc(h_w3.size() * 4); d_b3 = pool.alloc(128 * 4);
    d_dw3 = pool.alloc(h_w3.size() * 4); d_db3 = pool.alloc(128 * 4);
    d_w4 = pool.alloc(h_w4.size() * 4); d_b4 = pool.alloc(256 * 4);
    d_dw4 = pool.alloc(h_w4.size() * 4); d_db4 = pool.alloc(256 * 4);
    d_w5 = pool.alloc(h_w5.size() * 4); d_b5 = pool.alloc(3 * 4);
    d_dw5 = pool.alloc(h_w5.size() * 4); d_db5 = pool.alloc(3 * 4);
    
    // Double buffering for input
    float *d_input[2];
    d_input[0] = pool.alloc(s_in * 4);
    d_input[1] = pool.alloc(s_in * 4);
    
    // Forward buffers
    float *d_l1, *d_p1, *d_l2, *d_p2, *d_l3, *d_u3, *d_l4, *d_u4, *d_out;
    d_l1 = pool.alloc(s_l1 * 4); d_p1 = pool.alloc(s_p1 * 4);
    d_l2 = pool.alloc(s_l2 * 4); d_p2 = pool.alloc(s_p2 * 4);
    d_l3 = pool.alloc(s_l3 * 4); d_u3 = pool.alloc(s_u3 * 4);
    d_l4 = pool.alloc(s_l4 * 4); d_u4 = pool.alloc(s_u4 * 4);
    d_out = pool.alloc(s_in * 4);
    
    // Im2col buffers
    float *d_col1, *d_col2, *d_col3, *d_col4, *d_col5;
    d_col1 = pool.alloc(col1_size * 4);
    d_col2 = pool.alloc(col2_size * 4);
    d_col3 = pool.alloc(col3_size * 4);
    d_col4 = pool.alloc(col4_size * 4);
    d_col5 = pool.alloc(col5_size * 4);
    
    // Backward buffers
    float *d_dl1, *d_dp1, *d_dl2, *d_dp2, *d_dl3, *d_du3, *d_dl4, *d_du4, *d_dout;
    float *d_dcol;
    d_dl1 = pool.alloc(s_l1 * 4); d_dp1 = pool.alloc(s_p1 * 4);
    d_dl2 = pool.alloc(s_l2 * 4); d_dp2 = pool.alloc(s_p2 * 4);
    d_dl3 = pool.alloc(s_l3 * 4); d_du3 = pool.alloc(s_u3 * 4);
    d_dl4 = pool.alloc(s_l4 * 4); d_du4 = pool.alloc(s_u4 * 4);
    d_dout = pool.alloc(s_in * 4);
    d_dcol = pool.alloc(col5_size * 4);
    
    int *d_idx1 = (int*)pool.alloc(s_p1 * 4);
    int *d_idx2 = (int*)pool.alloc(s_p2 * 4);
    
    float* d_loss = pool.alloc(4);
    
    std::cout << "Memory: " << pool.get_total() / (1024 * 1024) << " MB\n\n";
    
    // Pinned host memory
    float* h_pinned_input;
    cudaMallocHost(&h_pinned_input, s_in * 4);
    
    // Copy weights
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1.data(), 256 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2.data(), 128 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3.data(), h_w3.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3.data(), 128 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4, h_w4.data(), h_w4.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4, h_b4.data(), 256 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5, h_w5.data(), h_w5.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5, h_b5.data(), 3 * 4, cudaMemcpyHostToDevice);
    
    int num_batches = dataset.get_num_train() / B;
    std::cout << "Training: " << EPOCHS << " epochs, " << num_batches << " batches\n\n";
    
    // Create streams
    cudaStream_t stream_compute, stream_transfer;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_transfer);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto ep_start = std::chrono::high_resolution_clock::now();
        
        cudaMemsetAsync(d_loss, 0, 4, stream_compute);
        
        // Pre-load first batch
        memcpy(h_pinned_input, dataset.get_train_images_ptr(), s_in * 4);
        cudaMemcpyAsync(d_input[0], h_pinned_input, s_in * 4, cudaMemcpyHostToDevice, stream_transfer);
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int curr_buf = batch % 2;
            int next_buf = (batch + 1) % 2;
            float* curr_input = d_input[curr_buf];
            
            // Async load next batch
            if (batch + 1 < num_batches) {
                cudaStreamSynchronize(stream_transfer);
                memcpy(h_pinned_input, dataset.get_train_images_ptr() + (batch + 1) * s_in, s_in * 4);
                cudaMemcpyAsync(d_input[next_buf], h_pinned_input, s_in * 4, 
                               cudaMemcpyHostToDevice, stream_transfer);
            }
            
            if (batch == 0) cudaStreamSynchronize(stream_transfer);
            
            // ========== FORWARD (Fused GEMM+Bias+ReLU) ==========
            // Layer 1: Conv + ReLU + MaxPool
            im2col_kernel<<<GRID(col1_size), BLOCK_SIZE, 0, stream_compute>>>(
                curr_input, d_col1, B, 32, 32, 3, 3, 1, 32, 32);
            gemm_nt_bias_relu(d_col1, d_w1, d_b1, d_l1, B * 32 * 32, 3 * 9, 256, true, stream_compute);
            maxpool_kernel<<<GRID(s_p1), BLOCK_SIZE, 0, stream_compute>>>(d_l1, d_p1, d_idx1, B, 32, 32, 256);
            
            // Layer 2: Conv + ReLU + MaxPool
            im2col_kernel<<<GRID(col2_size), BLOCK_SIZE, 0, stream_compute>>>(
                d_p1, d_col2, B, 16, 16, 256, 3, 1, 16, 16);
            gemm_nt_bias_relu(d_col2, d_w2, d_b2, d_l2, B * 16 * 16, 256 * 9, 128, true, stream_compute);
            maxpool_kernel<<<GRID(s_p2), BLOCK_SIZE, 0, stream_compute>>>(d_l2, d_p2, d_idx2, B, 16, 16, 128);
            
            // Layer 3: Conv + ReLU + Upsample
            im2col_kernel<<<GRID(col3_size), BLOCK_SIZE, 0, stream_compute>>>(
                d_p2, d_col3, B, 8, 8, 128, 3, 1, 8, 8);
            gemm_nt_bias_relu(d_col3, d_w3, d_b3, d_l3, B * 8 * 8, 128 * 9, 128, true, stream_compute);
            upsample_kernel<<<GRID(s_u3), BLOCK_SIZE, 0, stream_compute>>>(d_l3, d_u3, B, 8, 8, 128);
            
            // Layer 4: Conv + ReLU + Upsample
            im2col_kernel<<<GRID(col4_size), BLOCK_SIZE, 0, stream_compute>>>(
                d_u3, d_col4, B, 16, 16, 128, 3, 1, 16, 16);
            gemm_nt_bias_relu(d_col4, d_w4, d_b4, d_l4, B * 16 * 16, 128 * 9, 256, true, stream_compute);
            upsample_kernel<<<GRID(s_u4), BLOCK_SIZE, 0, stream_compute>>>(d_l4, d_u4, B, 16, 16, 256);
            
            // Layer 5: Conv (no ReLU)
            im2col_kernel<<<GRID(col5_size), BLOCK_SIZE, 0, stream_compute>>>(
                d_u4, d_col5, B, 32, 32, 256, 3, 1, 32, 32);
            gemm_nt_bias_relu(d_col5, d_w5, d_b5, d_out, B * 32 * 32, 256 * 9, 3, false, stream_compute);
            
            // ========== FUSED LOSS + BACKWARD ==========
            mse_loss_backward_fused_kernel<<<256, 256, 0, stream_compute>>>(
                d_out, curr_input, d_dout, d_loss, s_in);
            
            // ========== BACKWARD WITH FUSED KERNELS ==========
            
            // Layer 5 backward (no ReLU - use standard kernels)
            gemm_nn(d_dout, d_w5, d_dcol, B * 32 * 32, 3, 256 * 9, stream_compute);
            col2im_kernel<<<GRID(s_u4), BLOCK_SIZE, 0, stream_compute>>>(
                d_dcol, d_du4, B, 32, 32, 256, 3, 1, 32, 32);
            gemm_tn(d_dout, d_col5, d_dw5, 3, B * 32 * 32, 256 * 9, stream_compute);
            bias_backward_kernel<<<3, BLOCK_SIZE, 0, stream_compute>>>(d_dout, d_db5, B * 32 * 32, 3);
            
            // Layer 4 backward (FUSED: upsample + relu backward)
            fused_upsample_relu_backward(d_du4, d_l4, d_dl4, B, 16, 16, 256, stream_compute);
            gemm_nn(d_dl4, d_w4, d_dcol, B * 16 * 16, 256, 128 * 9, stream_compute);
            col2im_kernel<<<GRID(s_u3), BLOCK_SIZE, 0, stream_compute>>>(
                d_dcol, d_du3, B, 16, 16, 128, 3, 1, 16, 16);
            gemm_tn(d_dl4, d_col4, d_dw4, 256, B * 16 * 16, 128 * 9, stream_compute);
            bias_backward_kernel<<<256, BLOCK_SIZE, 0, stream_compute>>>(d_dl4, d_db4, B * 16 * 16, 256);
            
            // Layer 3 backward (FUSED: upsample + relu backward)
            fused_upsample_relu_backward(d_du3, d_l3, d_dl3, B, 8, 8, 128, stream_compute);
            gemm_nn(d_dl3, d_w3, d_dcol, B * 8 * 8, 128, 128 * 9, stream_compute);
            col2im_kernel<<<GRID(s_p2), BLOCK_SIZE, 0, stream_compute>>>(
                d_dcol, d_dp2, B, 8, 8, 128, 3, 1, 8, 8);
            gemm_tn(d_dl3, d_col3, d_dw3, 128, B * 8 * 8, 128 * 9, stream_compute);
            bias_backward_kernel<<<128, BLOCK_SIZE, 0, stream_compute>>>(d_dl3, d_db3, B * 8 * 8, 128);
            
            // Layer 2 backward (FUSED: zero + maxpool + relu backward)
            fused_maxpool_relu_backward(d_dp2, d_idx2, d_l2, d_dl2, s_p2, s_l2, stream_compute);
            gemm_nn(d_dl2, d_w2, d_dcol, B * 16 * 16, 128, 256 * 9, stream_compute);
            col2im_kernel<<<GRID(s_p1), BLOCK_SIZE, 0, stream_compute>>>(
                d_dcol, d_dp1, B, 16, 16, 256, 3, 1, 16, 16);
            gemm_tn(d_dl2, d_col2, d_dw2, 128, B * 16 * 16, 256 * 9, stream_compute);
            bias_backward_kernel<<<128, BLOCK_SIZE, 0, stream_compute>>>(d_dl2, d_db2, B * 16 * 16, 128);
            
            // Layer 1 backward (FUSED: zero + maxpool + relu backward)
            fused_maxpool_relu_backward(d_dp1, d_idx1, d_l1, d_dl1, s_p1, s_l1, stream_compute);
            gemm_tn(d_dl1, d_col1, d_dw1, 256, B * 32 * 32, 3 * 9, stream_compute);
            bias_backward_kernel<<<256, BLOCK_SIZE, 0, stream_compute>>>(d_dl1, d_db1, B * 32 * 32, 256);
            
            // ========== SGD UPDATE (Vectorized) ==========
            sgd_vectorized_kernel<<<GRID(h_w1.size() / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_w1, d_dw1, h_w1.size(), LR);
            sgd_vectorized_kernel<<<GRID(256 / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_b1, d_db1, 256, LR);
            sgd_vectorized_kernel<<<GRID(h_w2.size() / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_w2, d_dw2, h_w2.size(), LR);
            sgd_vectorized_kernel<<<GRID(128 / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_b2, d_db2, 128, LR);
            sgd_vectorized_kernel<<<GRID(h_w3.size() / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_w3, d_dw3, h_w3.size(), LR);
            sgd_vectorized_kernel<<<GRID(128 / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_b3, d_db3, 128, LR);
            sgd_vectorized_kernel<<<GRID(h_w4.size() / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_w4, d_dw4, h_w4.size(), LR);
            sgd_vectorized_kernel<<<GRID(256 / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_b4, d_db4, 256, LR);
            sgd_vectorized_kernel<<<GRID(h_w5.size() / 4), BLOCK_SIZE, 0, stream_compute>>>(
                d_w5, d_dw5, h_w5.size(), LR);
            sgd_kernel<<<GRID(3), BLOCK_SIZE, 0, stream_compute>>>(d_b5, d_db5, 3, LR);
        }
        
        float h_loss;
        cudaMemcpyAsync(&h_loss, d_loss, 4, cudaMemcpyDeviceToHost, stream_compute);
        cudaStreamSynchronize(stream_compute);
        
        auto ep_end = std::chrono::high_resolution_clock::now();
        double ep_time = std::chrono::duration<double>(ep_end - ep_start).count();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(6) << h_loss / (num_batches * s_in)
                  << " | Time: " << std::setprecision(2) << ep_time << "s"
                  << " | " << std::setprecision(0) << (num_batches * B) / ep_time << " img/s\n";
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    
    // Save weights
    cudaMemcpy(h_w1.data(), d_w1, h_w1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1.data(), d_b1, 256 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w2.data(), d_w2, h_w2.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2.data(), d_b2, 128 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w3.data(), d_w3, h_w3.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b3.data(), d_b3, 128 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w4.data(), d_w4, h_w4.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b4.data(), d_b4, 256 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w5.data(), d_w5, h_w5.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b5.data(), d_b5, 3 * 4, cudaMemcpyDeviceToHost);
    
    system("mkdir -p ../weights");
    save_weights("../weights/enc_w1.bin", h_w1); save_weights("../weights/enc_b1.bin", h_b1);
    save_weights("../weights/enc_w2.bin", h_w2); save_weights("../weights/enc_b2.bin", h_b2);
    save_weights("../weights/dec_w3.bin", h_w3); save_weights("../weights/dec_b3.bin", h_b3);
    save_weights("../weights/dec_w4.bin", h_w4); save_weights("../weights/dec_b4.bin", h_b4);
    save_weights("../weights/dec_w5.bin", h_w5); save_weights("../weights/dec_b5.bin", h_b5);
    
    cudaFreeHost(h_pinned_input);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
    std::cout << "Saved weights.\n";
    return 0;
}