#ifndef OPTIMIZE_KERNEL_H
#define OPTIMIZE_KERNEL_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

// ============== CONSTANTS ==============
#define TILE_SIZE 16
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 8
#define BLOCK_SIZE 256
#define GRID(n) ((n + BLOCK_SIZE - 1) / BLOCK_SIZE)

// ============== MEMORY POOL ==============
class MemoryPool {
    std::vector<std::pair<float*, size_t>> buffers;
    size_t total = 0;
public:
    float* alloc(size_t bytes);
    size_t get_total() const;
    ~MemoryPool();
};

// ============== FUSED GEMM + BIAS + RELU ==============
void gemm_nt_bias_relu(const float* A, const float* B, const float* bias,
                       float* C, int M, int K, int N, bool relu = true, 
                       cudaStream_t stream = 0);

// ============== STANDARD GEMM ==============
void gemm_nn(const float* A, const float* B, float* C, 
             int M, int K, int N, cudaStream_t stream = 0);

void gemm_tn(const float* A, const float* B, float* C, 
             int M, int K, int N, cudaStream_t stream = 0);

// ============== IM2COL / COL2IM ==============
void im2col(const float* input, float* col,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, 
            cudaStream_t stream = 0);

void col2im(const float* col, float* input_grad,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, 
            cudaStream_t stream = 0);

// ============== POOLING ==============
void maxpool_forward(const float* input, float* output, int* indices,
                     int B, int H_in, int W_in, int C, 
                     cudaStream_t stream = 0);

void upsample_forward(const float* input, float* output,
                      int B, int H_in, int W_in, int C, 
                      cudaStream_t stream = 0);

// ============== FUSED BACKWARD OPERATIONS ==============
void mse_loss_backward_fused(const float* pred, const float* target,
                             float* grad, float* partial_loss,
                             int size, cudaStream_t stream = 0);

void fused_upsample_relu_backward(const float* d_out, const float* fwd, 
                                  float* d_in, int B, int H_in, int W_in, int C, 
                                  cudaStream_t stream = 0);

void fused_maxpool_relu_backward(const float* d_out, const int* indices, 
                                 const float* fwd, float* d_in, 
                                 int pool_size, int input_size, 
                                 cudaStream_t stream = 0);

void gemm_nn_relu_backward(const float* d_out, const float* fwd, 
                           const float* W, float* d_col, 
                           int M, int K, int N, cudaStream_t stream = 0);

void gemm_tn_relu_backward(const float* d_out, const float* fwd, 
                           const float* col, float* dW, 
                           int M, int K, int N, cudaStream_t stream = 0);

// ============== NON-FUSED BACKWARD OPERATIONS ==============
void relu_backward(const float* d_out, const float* fwd, float* d_in,
                   int size, cudaStream_t stream = 0);

void maxpool_backward(const float* d_out, const int* indices, float* d_in,
                      int size, cudaStream_t stream = 0);

void upsample_backward(const float* d_out, float* d_in,
                       int B, int H_in, int W_in, int C, 
                       cudaStream_t stream = 0);

void bias_backward(const float* d_out, float* d_bias,
                   int B_HW, int C, cudaStream_t stream = 0);

void bias_backward_relu(const float* d_out, const float* fwd, float* d_bias,
                        int B_HW, int C, cudaStream_t stream = 0);

// ============== VECTORIZED UTILITIES ==============
void fill_zeros_vectorized(float* data, int size, cudaStream_t stream = 0);

void sgd_update_vectorized(float* weights, const float* gradients,
                           int size, float learning_rate, 
                           cudaStream_t stream = 0);

void sgd_update(float* weights, const float* gradients,
                int size, float learning_rate, 
                cudaStream_t stream = 0);

// ============== INITIALIZATION & I/O ==============
void init_random(std::vector<float>& v, int fan_in, int fan_out);

void save_weights(const std::string& filename, const std::vector<float>& data);

bool load_weights(const std::string& filename, std::vector<float>& data);

#endif // OPTIMIZE_KERNEL_H