#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// --- CUDA ERROR CHECKING MACRO ---
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// --- DEVICE-SIDE HELPER FUNCTIONS ---

// Function executed on the GPU to calculate the index.
__device__ inline int get_idx_dev(int b, int h, int w, int c, int H, int W, int C) {
    return b * (H * W * C) + h * (W * C) + w * C + c;
}

// Helper to zero out memory (Crucial for backward passes that use atomicAdd or accumulation)
__global__ void fill_zeros(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 0.0f;
}

// --- KERNEL LAUNCH CONFIGURATION ---
dim3 get_1d_dims(int total_size) {
    const int THREADS_PER_BLOCK = 256;
    int blocks = (total_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return dim3(blocks, 1, 1);
}

// ====================================================================
//                             1. CONVOLUTION
// ====================================================================

// --- FORWARD KERNEL ---
__global__ void conv2d_kernel(float* input, float* weight, float* bias, float* output, ConvParam p) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_size = p.batch * p.out_h * p.out_w * p.out_c;

    if (out_idx < total_output_size) {
        int C = p.out_c;
        int W = p.out_w;
        int H = p.out_h;

        int oc = out_idx % C;
        int temp = out_idx / C;
        int ow = temp % W;
        temp = temp / W;
        int oh = temp % H;
        int b = temp / H;

        float sum = bias[oc];

        for (int ic = 0; ic < p.in_c; ++ic) {
            for (int kh = 0; kh < p.k_size; ++kh) {
                for (int kw = 0; kw < p.k_size; ++kw) {
                    int ih = oh * p.stride - p.padding + kh;
                    int iw = ow * p.stride - p.padding + kw;

                    if (ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w) {
                        int in_idx = get_idx_dev(b, ih, iw, ic, p.in_h, p.in_w, p.in_c);
                        
                        // Weight layout: [out_c][in_c][k][k]
                        int w_idx = oc * (p.in_c * p.k_size * p.k_size) 
                                  + ic * (p.k_size * p.k_size) 
                                  + kh * p.k_size + kw;

                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[out_idx] = sum;
    }
}

// --- BACKWARD KERNELS ---

// 1. Calculate Gradients w.r.t Input (d_input)
// This effectively performs a "transposed convolution" or "deconvolution" logic
__global__ void conv2d_backward_input_kernel(float* d_output, float* weight, float* d_input, ConvParam p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_in_size = p.batch * p.in_h * p.in_w * p.in_c;

    if (idx < total_in_size) {
        int c = idx % p.in_c;
        int temp = idx / p.in_c;
        int w = temp % p.in_w;
        temp = temp / p.in_w;
        int h = temp % p.in_h;
        int b = temp / p.in_h;

        float sum = 0.0f;

        // Iterate over output channels and kernel window
        for (int oc = 0; oc < p.out_c; ++oc) {
            for (int kh = 0; kh < p.k_size; ++kh) {
                for (int kw = 0; kw < p.k_size; ++kw) {
                    // Logic to find the output pixel that this input pixel contributed to
                    int h_shifted = h + p.padding - kh;
                    int w_shifted = w + p.padding - kw;

                    if (h_shifted % p.stride == 0 && w_shifted % p.stride == 0) {
                        int oh = h_shifted / p.stride;
                        int ow = w_shifted / p.stride;

                        if (oh >= 0 && oh < p.out_h && ow >= 0 && ow < p.out_w) {
                            int out_idx = get_idx_dev(b, oh, ow, oc, p.out_h, p.out_w, p.out_c);
                            int w_idx = oc * (p.in_c * p.k_size * p.k_size) 
                                      + c * (p.k_size * p.k_size) 
                                      + kh * p.k_size + kw;
                            sum += d_output[out_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        d_input[idx] = sum;
    }
}

// 2. Calculate Gradients w.r.t Weights (d_weight)
__global__ void conv2d_backward_weight_kernel(float* d_output, float* input, float* d_weight, ConvParam p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = p.out_c * p.in_c * p.k_size * p.k_size;

    if (idx < total_weights) {
        int kw = idx % p.k_size;
        int temp = idx / p.k_size;
        int kh = temp % p.k_size;
        temp = temp / p.k_size;
        int ic = temp % p.in_c;
        int oc = temp / p.in_c;

        float sum = 0.0f;

        // Sum gradients over the entire batch and image spatial dimensions
        for (int b = 0; b < p.batch; ++b) {
            for (int oh = 0; oh < p.out_h; ++oh) {
                for (int ow = 0; ow < p.out_w; ++ow) {
                    int ih = oh * p.stride - p.padding + kh;
                    int iw = ow * p.stride - p.padding + kw;

                    if (ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w) {
                        int in_idx = get_idx_dev(b, ih, iw, ic, p.in_h, p.in_w, p.in_c);
                        int out_idx = get_idx_dev(b, oh, ow, oc, p.out_h, p.out_w, p.out_c);
                        sum += input[in_idx] * d_output[out_idx];
                    }
                }
            }
        }
        d_weight[idx] = sum;
    }
}

// 3. Calculate Gradients w.r.t Bias (d_bias)
__global__ void conv2d_backward_bias_kernel(float* d_output, float* d_bias, ConvParam p) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc < p.out_c) {
        float sum = 0.0f;
        for (int b = 0; b < p.batch; ++b) {
            for (int h = 0; h < p.out_h; ++h) {
                for (int w = 0; w < p.out_w; ++w) {
                    int out_idx = get_idx_dev(b, h, w, oc, p.out_h, p.out_w, p.out_c);
                    sum += d_output[out_idx];
                }
            }
        }
        d_bias[oc] = sum;
    }
}

// --- HOST WRAPPERS ---
void conv2d(float* input, float* weight, float* bias, float* output, ConvParam p) {
    int total_output_size = p.batch * p.out_h * p.out_w * p.out_c;
    conv2d_kernel<<<get_1d_dims(total_output_size), 256>>>(input, weight, bias, output, p);
    checkCudaErrors(cudaGetLastError());
}

void conv2d_backward(float* d_output, float* input, float* weight, 
                     float* d_input, float* d_weight, float* d_bias, ConvParam p) {
    
    // 1. Calculate d_input
    int in_size = p.batch * p.in_h * p.in_w * p.in_c;
    fill_zeros<<<get_1d_dims(in_size), 256>>>(d_input, in_size); // Clear memory first
    conv2d_backward_input_kernel<<<get_1d_dims(in_size), 256>>>(d_output, weight, d_input, p);
    checkCudaErrors(cudaGetLastError());

    // 2. Calculate d_weight
    int w_size = p.out_c * p.in_c * p.k_size * p.k_size;
    // Note: d_weight does not need clearing as the kernel overwrites it (assignment, not accumulation)
    conv2d_backward_weight_kernel<<<get_1d_dims(w_size), 256>>>(d_output, input, d_weight, p);
    checkCudaErrors(cudaGetLastError());

    // 3. Calculate d_bias
    conv2d_backward_bias_kernel<<<get_1d_dims(p.out_c), 256>>>(d_output, d_bias, p);
    checkCudaErrors(cudaGetLastError());
}



// ====================================================================
//                             2. ReLU ACTIVATION
// ====================================================================

__global__ void relu_kernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = (data[i] < 0.0f) ? 0.0f : data[i];
    }
}

__global__ void relu_backward_kernel(float* d_output, float* input, float* d_input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_input[i] = (input[i] > 0) ? d_output[i] : 0.0f;
    }
}

void relu(float* data, int size) {
    relu_kernel<<<get_1d_dims(size), 256>>>(data, size);
    checkCudaErrors(cudaGetLastError());
}

void relu_backward(float* d_output, float* input, float* d_input, int size) {
    relu_backward_kernel<<<get_1d_dims(size), 256>>>(d_output, input, d_input, size);
    checkCudaErrors(cudaGetLastError());
}

// ====================================================================
//                             3. MAX POOLING
// ====================================================================

// --- FORWARD KERNEL ---
__global__ void maxpool_kernel(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int total_output_size = batch * out_h * out_w * in_c;
    int stride = 2;

    if (out_idx < total_output_size) {
        int C = in_c;
        int W = out_w;
        int H = out_h;

        int c = out_idx % C;
        int temp = out_idx / C;
        int ow = temp % W;
        temp = temp / W;
        int oh = temp % H;
        int b = temp / H;

        float max_val = -1e9;
        
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                int in_idx = get_idx_dev(b, ih, iw, c, in_h, in_w, in_c);
                if (input[in_idx] > max_val) max_val = input[in_idx];
            }
        }
        output[out_idx] = max_val;
    }
}

// --- BACKWARD KERNEL ---
__global__ void maxpool_backward_kernel(float* d_output, float* input, float* d_input, 
                                        int batch, int in_h, int in_w, int in_c) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int total_output = batch * out_h * out_w * in_c;

    if (out_idx < total_output) {
        int c = out_idx % in_c;
        int temp = out_idx / in_c;
        int ow = temp % out_w;
        temp = temp / out_w;
        int oh = temp % out_h;
        int b = temp / out_h;

        int start_h = oh * 2;
        int start_w = ow * 2;
        float max_val = -1e9;
        int max_idx = -1;

        // Re-find the max value position
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                int ih = start_h + kh;
                int iw = start_w + kw;
                int in_idx = get_idx_dev(b, ih, iw, c, in_h, in_w, in_c);
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_idx;
                }
            }
        }

        // Atomic add the gradient to the winner pixel
        if (max_idx != -1) {
            atomicAdd(&d_input[max_idx], d_output[out_idx]);
        }
    }
}

void maxpool(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int total_output_size = batch * out_h * out_w * in_c;
    maxpool_kernel<<<get_1d_dims(total_output_size), 256>>>(input, output, batch, in_h, in_w, in_c);
    checkCudaErrors(cudaGetLastError());
}

void maxpool_backward(float* d_output, float* input, float* d_input, 
                      int batch, int in_h, int in_w, int in_c) {
    int size_input = batch * in_h * in_w * in_c;
    fill_zeros<<<get_1d_dims(size_input), 256>>>(d_input, size_input); // Must clear accumulator

    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int size_output = batch * out_h * out_w * in_c;
    maxpool_backward_kernel<<<get_1d_dims(size_output), 256>>>(d_output, input, d_input, batch, in_h, in_w, in_c);
    checkCudaErrors(cudaGetLastError());
}

// ====================================================================
//                             4. UPSAMPLE
// ====================================================================

// --- FORWARD KERNEL ---
__global__ void upsample_kernel(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    int total_output_size = batch * out_h * out_w * in_c;

    if (out_idx < total_output_size) {
        int C = in_c;
        int W = out_w;
        int H = out_h;

        int c = out_idx % C;
        int temp = out_idx / C;
        int ow = temp % W;
        temp = temp / W;
        int oh = temp % H;
        int b = temp / H;

        int ih = oh / 2;
        int iw = ow / 2;
        int in_idx = get_idx_dev(b, ih, iw, c, in_h, in_w, in_c);
        output[out_idx] = input[in_idx];
    }
}

// --- BACKWARD KERNEL ---
__global__ void upsample_backward_kernel(float* d_output, float* d_input, 
                                         int batch, int in_h, int in_w, int in_c) {
    // Note: in_h/in_w here refer to the input of the forward pass (small image)
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    int total_output_size = batch * out_h * out_w * in_c;

    if (out_idx < total_output_size) {
        int C = in_c;
        int W = out_w;
        int H = out_h;

        int c = out_idx % C;
        int temp = out_idx / C;
        int ow = temp % W;
        temp = temp / W;
        int oh = temp % H;
        int b = temp / H;

        // Map larger image pixel back to small image pixel
        int ih = oh / 2;
        int iw = ow / 2;
        int in_idx = get_idx_dev(b, ih, iw, c, in_h, in_w, in_c);

        // Atomic add required as 4 output pixels map to 1 input pixel
        atomicAdd(&d_input[in_idx], d_output[out_idx]);
    }
}

void upsample(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    int total_output_size = batch * out_h * out_w * in_c;
    upsample_kernel<<<get_1d_dims(total_output_size), 256>>>(input, output, batch, in_h, in_w, in_c);
    checkCudaErrors(cudaGetLastError());
}

void upsample_backward(float* d_output, float* d_input, int batch, int in_h, int in_w, int in_c) {
    int size_input = batch * in_h * in_w * in_c;
    fill_zeros<<<get_1d_dims(size_input), 256>>>(d_input, size_input); // Clear accumulator

    int out_h = in_h * 2;
    int out_w = in_w * 2;
    int size_output = batch * out_h * out_w * in_c;
    upsample_backward_kernel<<<get_1d_dims(size_output), 256>>>(d_output, d_input, batch, in_h, in_w, in_c);
    checkCudaErrors(cudaGetLastError());
}

// ====================================================================
//                             5. MSE LOSS
// ====================================================================

__global__ void mse_diff_kernel(float* pred, float* target, float* diff_sq, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];
        diff_sq[i] = diff * diff;
    }
}

float mse_loss(float* pred, float* target, int size) {
    float* diff_sq_d;
    checkCudaErrors(cudaMalloc((void**)&diff_sq_d, size * sizeof(float)));

    mse_diff_kernel<<<get_1d_dims(size), 256>>>(pred, target, diff_sq_d, size);
    checkCudaErrors(cudaGetLastError());

    // NOTE: For performance, this reduction should be done on GPU.
    float* diff_sq_h = (float*)malloc(size * sizeof(float));
    checkCudaErrors(cudaMemcpy(diff_sq_h, diff_sq_d, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += diff_sq_h[i];
    }

    checkCudaErrors(cudaFree(diff_sq_d));
    free(diff_sq_h);
    
    return sum / size;
}

// ====================================================================
//                             6. OPTIMIZER
// ====================================================================

__global__ void update_weights_kernel(float* weights, float* d_weights, int size, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        weights[i] = weights[i] - lr * d_weights[i];
    }
}

void update_weights(float* weights, float* d_weights, int size, float lr) {
    update_weights_kernel<<<get_1d_dims(size), 256>>>(weights, d_weights, size, lr);
    checkCudaErrors(cudaGetLastError());
}