#ifndef TRAIN_GPU_OPTIMIZE_H
#define TRAIN_GPU_OPTIMIZE_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

// ============== CONSTANTS ==============
#define BLOCK_SIZE 256
#define TILE_SIZE 32
#define GRID(n) ((n + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Optimized GEMM tile sizes
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 4
#define THREAD_N 4

// ============== CUDA ERROR CHECKING ==============
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err); \
    } \
} while(0)

// ============== MEMORY POOL CLASS ==============
class MemoryPool {
public:
    float* alloc(size_t bytes);
    size_t get_total() const;
    ~MemoryPool();

private:
    std::vector<std::pair<float*, size_t>> buffers;
    size_t total = 0;
};

// ============== GEMM OPERATIONS ==============

// Fused GEMM + Bias + ReLU: C[M,N] = ReLU(A[M,K] * B^T[N,K] + bias[N])
void gemm_nt_bias_relu(const float* A, const float* B, const float* bias,
                       float* C, int M, int K, int N, bool relu, cudaStream_t stream);

// Standard GEMM: C[M,N] = A[M,K] * B[K,N]
void gemm_nn(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream);

// Transposed GEMM: C[M,N] = A^T[K,M] * B[K,N]
void gemm_tn(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream);

// ============== CONVOLUTION HELPERS ==============

// Im2col kernel launcher
void im2col(const float* input, float* col,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, cudaStream_t stream);

// Col2im kernel launcher
void col2im(const float* col, float* input_grad,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, cudaStream_t stream);

// ============== POOLING OPERATIONS ==============

// MaxPool forward
void maxpool_forward(const float* input, float* output, int* indices,
                     int B, int H_in, int W_in, int C, cudaStream_t stream);

// Upsample forward (nearest neighbor 2x)
void upsample_forward(const float* input, float* output,
                      int B, int H_in, int W_in, int C, cudaStream_t stream);

// ============== FUSED BACKWARD OPERATIONS ==============

// Fused MSE loss computation + backward gradient
void mse_loss_backward_fused(const float* pred, const float* target,
                             float* grad, float* partial_loss,
                             int size, cudaStream_t stream);

// Fused upsample + ReLU backward
void fused_upsample_relu_backward(const float* d_out, const float* fwd, float* d_in,
                                  int B, int H_in, int W_in, int C, cudaStream_t stream);

// Fused maxpool + ReLU backward
void fused_maxpool_relu_backward(const float* d_out, const int* indices, const float* fwd,
                                 float* d_in, int pool_size, int input_size, cudaStream_t stream);

// Fused GEMM_NN + ReLU backward (for input gradient)
void gemm_nn_relu_backward(const float* d_out, const float* fwd, const float* W,
                           float* d_col, int M, int K, int N, cudaStream_t stream);

// Fused GEMM_TN + ReLU backward (for weight gradient)
void gemm_tn_relu_backward(const float* d_out, const float* fwd, const float* col,
                           float* dW, int M, int K, int N, cudaStream_t stream);

// ============== NON-FUSED BACKWARD OPERATIONS ==============

// ReLU backward
void relu_backward(const float* d_out, const float* fwd, float* d_in,
                   int size, cudaStream_t stream);

// MaxPool backward
void maxpool_backward(const float* d_out, const int* indices, float* d_in,
                      int size, cudaStream_t stream);

// Upsample backward
void upsample_backward(const float* d_out, float* d_in,
                       int B, int H_in, int W_in, int C, cudaStream_t stream);

// Bias backward
void bias_backward(const float* d_out, float* d_bias,
                   int B_HW, int C, cudaStream_t stream);

// Bias backward with fused ReLU
void bias_backward_relu(const float* d_out, const float* fwd, float* d_bias,
                        int B_HW, int C, cudaStream_t stream);

// ============== OPTIMIZER OPERATIONS ==============

// SGD update (vectorized)
void sgd_update_vectorized(float* weights, const float* gradients,
                           int size, float learning_rate, cudaStream_t stream);

// SGD update (standard)
void sgd_update(float* weights, const float* gradients,
                int size, float learning_rate, cudaStream_t stream);

// ============== UTILITY OPERATIONS ==============

// Fill tensor with zeros (vectorized)
void fill_zeros_vectorized(float* data, int size, cudaStream_t stream);

// ============== WEIGHT INITIALIZATION ==============

// Xavier/Glorot initialization
void init_random(std::vector<float>& weights, int fan_in, int fan_out);

// ============== I/O OPERATIONS ==============

// Save weights to binary file
void save_weights(const std::string& filename, const std::vector<float>& data);

// Load weights from binary file
bool load_weights(const std::string& filename, std::vector<float>& data);

#endif // TRAIN_GPU_OPTIMIZE_H