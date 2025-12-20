#ifndef OPTIMIZE_KERNEL_H
#define OPTIMIZE_KERNEL_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

// Block/Grid configuration
#define BLOCK_SIZE 256
#define GRID(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Tiling parameters
#define TILE_SIZE 16
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 8

// Memory pool
struct MemoryPool {
    std::vector<std::pair<float*, size_t>> buffers;
    size_t total = 0;
    float* alloc(size_t bytes);
    size_t get_total() const;
    ~MemoryPool();
};

// GEMM kernels (now with merged bias in weight matrix)
void gemm_nt_relu(const float* A, const float* W, float* C, 
                  int M, int K, int N, bool relu, cudaStream_t stream);
void gemm_nn(const float* A, const float* B, float* C, 
             int M, int K, int N, cudaStream_t stream);
void gemm_tn(const float* A, const float* B, float* C, 
             int M, int K, int N, cudaStream_t stream);

// Im2col/Col2im with bias augmentation
void im2col_with_bias(const float* input, float* col,
                      int B, int H, int W, int C,
                      int K, int P, int H_out, int W_out, cudaStream_t stream);
void col2im(const float* col, float* input_grad,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, cudaStream_t stream);

// Pooling
void maxpool_forward(const float* input, float* output, int* indices,
                     int B, int H_in, int W_in, int C, cudaStream_t stream);
void upsample_forward(const float* input, float* output,
                      int B, int H_in, int W_in, int C, cudaStream_t stream);

// Backward kernels
void mse_loss_backward_fused(const float* pred, const float* target,
                             float* grad, float* partial_loss,
                             int size, cudaStream_t stream);
void fused_upsample_relu_backward(const float* d_out, const float* fwd, float* d_in,
                                  int B, int H_in, int W_in, int C, cudaStream_t stream);
void fused_maxpool_relu_backward(const float* d_out, const int* indices, const float* fwd,
                                 float* d_in, int pool_size, int input_size, cudaStream_t stream);
void relu_backward(const float* d_out, const float* fwd, float* d_in,
                   int size, cudaStream_t stream);
void maxpool_backward(const float* d_out, const int* indices, float* d_in,
                      int size, cudaStream_t stream);
void upsample_backward(const float* d_out, float* d_in,
                       int B, int H_in, int W_in, int C, cudaStream_t stream);

// Utility kernels
void fill_zeros_vectorized(float* data, int size, cudaStream_t stream);
void sgd_update_vectorized(float* weights, const float* gradients,
                           int size, float learning_rate, cudaStream_t stream);
void sgd_update(float* weights, const float* gradients,
                int size, float learning_rate, cudaStream_t stream);

// Weight initialization
void init_random(std::vector<float>& v, int fan_in, int fan_out);

// I/O
void save_weights(const std::string& f, const std::vector<float>& d);
bool load_weights(const std::string& f, std::vector<float>& d);

#endif