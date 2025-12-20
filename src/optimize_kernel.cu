#include "optimize_kernel.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

// ============== MEMORY POOL IMPLEMENTATION ==============
float* MemoryPool::alloc(size_t bytes) {
    float* p;
    cudaMalloc(&p, bytes);
    buffers.push_back({p, bytes});
    total += bytes;
    return p;
}

size_t MemoryPool::get_total() const { return total; }

MemoryPool::~MemoryPool() {
    for (auto& b : buffers) cudaFree(b.first);
}

// ============== FUSED GEMM + BIAS + RELU KERNELS ==============

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

// ============== IM2COL / COL2IM KERNELS ==============

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

void im2col(const float* input, float* col,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, cudaStream_t stream) {
    int total = B * H_out * W_out * C * K * K;
    im2col_kernel<<<GRID(total), BLOCK_SIZE, 0, stream>>>(input, col, B, H, W, C, K, P, H_out, W_out);
}

void col2im(const float* col, float* input_grad,
            int B, int H, int W, int C,
            int K, int P, int H_out, int W_out, cudaStream_t stream) {
    int total = B * H * W * C;
    col2im_kernel<<<GRID(total), BLOCK_SIZE, 0, stream>>>(col, input_grad, B, H, W, C, K, P, H_out, W_out);
}

// ============== POOLING KERNELS ==============

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

void maxpool_forward(const float* input, float* output, int* indices,
                     int B, int H_in, int W_in, int C, cudaStream_t stream) {
    int H_out = H_in / 2, W_out = W_in / 2;
    int total = B * H_out * W_out * C;
    maxpool_kernel<<<GRID(total), BLOCK_SIZE, 0, stream>>>(input, output, indices, B, H_in, W_in, C);
}

void upsample_forward(const float* input, float* output,
                      int B, int H_in, int W_in, int C, cudaStream_t stream) {
    int H_out = H_in * 2, W_out = W_in * 2;
    int total = B * H_out * W_out * C;
    upsample_kernel<<<GRID(total), BLOCK_SIZE, 0, stream>>>(input, output, B, H_in, W_in, C);
}

// ============== FUSED BACKWARD KERNELS ==============

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

void mse_loss_backward_fused(const float* pred, const float* target,
                             float* grad, float* partial_loss,
                             int size, cudaStream_t stream) {
    mse_loss_backward_fused_kernel<<<256, 256, 0, stream>>>(pred, target, grad, partial_loss, size);
}

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
    
    float sum = d_out[b * (H_out * W_out * C) + ho * (W_out * C) + wo * C + c]
              + d_out[b * (H_out * W_out * C) + ho * (W_out * C) + (wo + 1) * C + c]
              + d_out[b * (H_out * W_out * C) + (ho + 1) * (W_out * C) + wo * C + c]
              + d_out[b * (H_out * W_out * C) + (ho + 1) * (W_out * C) + (wo + 1) * C + c];
    
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

void fill_zeros_vectorized(float* data, int size, cudaStream_t stream) {
    fill_zeros_vectorized_kernel<<<GRID(size / 4), BLOCK_SIZE, 0, stream>>>(data, size);
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

void sgd_update_vectorized(float* weights, const float* gradients,
                           int size, float learning_rate, cudaStream_t stream) {
    sgd_vectorized_kernel<<<GRID(size / 4), BLOCK_SIZE, 0, stream>>>(weights, gradients, size, learning_rate);
}

void sgd_update(float* weights, const float* gradients,
                int size, float learning_rate, cudaStream_t stream) {
    sgd_kernel<<<GRID(size), BLOCK_SIZE, 0, stream>>>(weights, gradients, size, learning_rate);
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
    
    if (fwd[target_idx] > 0.0f) {
        atomicAdd(&d_in[target_idx], grad);
    }
}

void fused_maxpool_relu_backward(const float* d_out, const int* indices, const float* fwd,
                                 float* d_in, int pool_size, int input_size, cudaStream_t stream) {
    fill_zeros_vectorized_kernel<<<GRID(input_size / 4), BLOCK_SIZE, 0, stream>>>(d_in, input_size);
    fused_maxpool_relu_backward_kernel<<<GRID(pool_size), BLOCK_SIZE, 0, stream>>>(
        d_out, indices, fwd, d_in, pool_size);
}

// ============== FUSED GEMM + RELU BACKWARD KERNELS ==============

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

// ============== NON-FUSED BACKWARD KERNELS ==============

__global__ void relu_backward_kernel(const float* d_out, const float* fwd, float* d_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_in[i] = (fwd[i] > 0.0f) ? d_out[i] : 0.0f;
}

void relu_backward(const float* d_out, const float* fwd, float* d_in,
                   int size, cudaStream_t stream) {
    relu_backward_kernel<<<GRID(size), BLOCK_SIZE, 0, stream>>>(d_out, fwd, d_in, size);
}

__global__ void maxpool_backward_kernel(const float* d_out, const int* idx, float* d_in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) atomicAdd(&d_in[idx[i]], d_out[i]);
}

void maxpool_backward(const float* d_out, const int* indices, float* d_in,
                      int size, cudaStream_t stream) {
    maxpool_backward_kernel<<<GRID(size), BLOCK_SIZE, 0, stream>>>(d_out, indices, d_in, size);
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

void upsample_backward(const float* d_out, float* d_in,
                       int B, int H_in, int W_in, int C, cudaStream_t stream) {
    int total = B * H_in * W_in * C;
    upsample_backward_kernel<<<GRID(total), BLOCK_SIZE, 0, stream>>>(d_out, d_in, B, H_in, W_in, C);
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

void bias_backward(const float* d_out, float* d_bias,
                   int B_HW, int C, cudaStream_t stream) {
    bias_backward_kernel<<<C, BLOCK_SIZE, 0, stream>>>(d_out, d_bias, B_HW, C);
}

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

void bias_backward_relu(const float* d_out, const float* fwd, float* d_bias,
                        int B_HW, int C, cudaStream_t stream) {
    bias_backward_relu_kernel<<<C, BLOCK_SIZE, 0, stream>>>(d_out, fwd, d_bias, B_HW, C);
}