#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>      // For sqrt
#include "cifar10_dataset.h"
#include "kernels.h"  // Assuming this includes ConvParam_G struct
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h> // For malloc/free
// Assuming ConvParam_G is defined in kernels.h or provided elsewhere.
// For the purpose of making the code compile, I'll define a placeholder:
struct ConvParam_G {
            int B, H_in, W_in, C_in;
            int H_out, W_out, C_out;
            int K, S, P;
};
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
            if (result) {
                fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
                exit(EXIT_FAILURE);
    }
}
// Utility for CUDA error checking
void checkCudaErrors(cudaError_t code) {
            if (code != cudaSuccess) {
                std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " (Code: " << code << ")\n";
                exit(code);
    }
}
// --- DEVICE-SIDE HELPER FUNCTIONS ---
// Function executed on the GPU to calculate the index.
__device__ inline int get_idx_dev(int b, int h, int w, int c, int H, int W, int C) {
            return b * (H * W * C) + h * (W * C) + w * C + c;
}
// Helper to zero out memory (Crucial for backward passes that use atomicAdd or accumulation)
__global__ void fill_zeros(float* data, size_t size) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) data[idx] = 0.0f;
}
// --- KERNEL LAUNCH CONFIGURATION ---
dim3 get_1d_dims(size_t total_size) {
            const int THREADS_PER_BLOCK = 256;
            // We cast total_size to int for division, assuming total_size fits within standard integer limits
            // Use size_t and long long to avoid potential overflow issues with int casting
            size_t blocks = (total_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            return dim3((unsigned int)blocks, 1, 1);
}
// ====================================================================
//                          1. CONVOLUTION
// ====================================================================
// --- FORWARD KERNEL (Field names corrected to match ConvParam_G) ---
__global__ void conv2d_kernel(float* input, float* weight, float* bias, float* output, ConvParam_G p) {
            int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_output_size = p.B * p.H_out * p.W_out * p.C_out;
            if (out_idx < total_output_size) {
                int C = p.C_out;
                int W = p.W_out;
                int H = p.H_out;
                int oc = out_idx % C;
                int temp = out_idx / C;
                int ow = temp % W;
                temp = temp / W;
                int oh = temp % H;
                int b = temp / H;
                float sum = bias[oc];
                // Iterate over input channels, kernel height, and width
                for (int ic = 0; ic < p.C_in; ++ic) {
                    for (int kh = 0; kh < p.K; ++kh) {
                        for (int kw = 0; kw < p.K; ++kw) {
                            // Calculate input indices (ih, iw)
                            int ih = oh * p.S - p.P + kh;
                            int iw = ow * p.S - p.P + kw;
                            if (ih >= 0 && ih < p.H_in && iw >= 0 && iw < p.W_in) {
                                int in_idx = get_idx_dev(b, ih, iw, ic, p.H_in, p.W_in, p.C_in);
                                
                                // Weight layout: [C_out][C_in][K][K]
                                int w_idx = oc * (p.C_in * p.K * p.K) 
                                          + ic * (p.K * p.K) 
                                          + kh * p.K + kw;
                                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
}
    output[out_idx] = sum;
    }
}
// --- BACKWARD KERNELS (Field names corrected) ---
// 1. Calculate Gradients w.r.t Input (d_input)
__global__ void conv2d_backward_input_kernel(float* d_output, float* weight, float* d_input, ConvParam_G p) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_in_size = p.B * p.H_in * p.W_in * p.C_in;
            if (idx < total_in_size) {
                int c = idx % p.C_in;
                int temp = idx / p.C_in;
                int w = temp % p.W_in;
                temp = temp / p.W_in;
                int h = temp % p.H_in;
                int b = temp / p.H_in;
                float sum = 0.0f;
                // Iterate over output channels and kernel window
                for (int oc = 0; oc < p.C_out; ++oc) {
                    for (int kh = 0; kh < p.K; ++kh) {
                        for (int kw = 0; kw < p.K; ++kw) {
                            // Logic to find the output pixel that this input pixel contributed to
                            // This is essentially reverse mapping the convolution indices.
                            int h_shifted = h + p.P - kh;
                            int w_shifted = w + p.P - kw;
                            if (h_shifted % p.S == 0 && w_shifted % p.S == 0) {
                                int oh = h_shifted / p.S;
                                int ow = w_shifted / p.S;
                                if (oh >= 0 && oh < p.H_out && ow >= 0 && ow < p.W_out) {
                                    int out_idx = get_idx_dev(b, oh, ow, oc, p.H_out, p.W_out, p.C_out);
                                    
                                    // Weight layout: [C_out][C_in][K][K]
                                    int w_idx = oc * (p.C_in * p.K * p.K) 
                                              + c * (p.K * p.K) 
                                              + kh * p.K + kw;
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
__global__ void conv2d_backward_weight_kernel(float* d_output, float* input, float* d_weight, ConvParam_G p) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_weights = p.C_out * p.C_in * p.K * p.K;
            if (idx < total_weights) {
                int kw = idx % p.K;
                int temp = idx / p.K;
                int kh = temp % p.K;
                temp = temp / p.K;
                int ic = temp % p.C_in;
                int oc = temp / p.C_in;
                float sum = 0.0f;
                // Sum gradients over the entire batch and image spatial dimensions
                for (int b = 0; b < p.B; ++b) {
                    for (int oh = 0; oh < p.H_out; ++oh) {
                        for (int ow = 0; ow < p.W_out; ++ow) {
                            int ih = oh * p.S - p.P + kh;
                            int iw = ow * p.S - p.P + kw;
                            if (ih >= 0 && ih < p.H_in && iw >= 0 && iw < p.W_in) {
                                int in_idx = get_idx_dev(b, ih, iw, ic, p.H_in, p.W_in, p.C_in);
                                int out_idx = get_idx_dev(b, oh, ow, oc, p.H_out, p.W_out, p.C_out);
                                sum += input[in_idx] * d_output[out_idx];
            }
        }
    }
}
    d_weight[idx] = sum;
    }
}
// 3. Calculate Gradients w.r.t Bias (d_bias)
__global__ void conv2d_backward_bias_kernel(float* d_output, float* d_bias, ConvParam_G p) {
            int oc = blockIdx.x * blockDim.x + threadIdx.x;
            if (oc < p.C_out) {
                float sum = 0.0f;
                for (int b = 0; b < p.B; ++b) {
                    for (int h = 0; h < p.H_out; ++h) {
                        for (int w = 0; w < p.W_out; ++w) {
                            int out_idx = get_idx_dev(b, h, w, oc, p.H_out, p.W_out, p.C_out);
                            sum += d_output[out_idx];
        }
    }
}
    d_bias[oc] = sum;
    }
}
// ====================================================================
//                          2. ReLU ACTIVATION
// ====================================================================
__global__ void relu_kernel(float* data, size_t size) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                data[i] = (data[i] < 0.0f) ? 0.0f : data[i];
    }
}
__global__ void relu_backward_kernel(float* d_output, float* input, float* d_input, size_t size) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                // dL/dx = dL/dy * dy/dx. dy/dx = 1 if x > 0, 0 otherwise.
                d_input[i] = (input[i] > 0) ? d_output[i] : 0.0f;
    }
}
// ====================================================================
//                          3. MAX POOLING
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
            // Only thread for an output gradient (d_output) needs to run
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
    // Atomic add the gradient to the winner pixel. d_input must be zeroed beforehand.
    if (max_idx != -1) {
                    atomicAdd(&d_input[max_idx], d_output[out_idx]);
}
    }
}
// ====================================================================
//                          4. UPSAMPLE
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
                // Atomic add required as 4 output pixels map to 1 input pixel. d_input must be zeroed beforehand.
                atomicAdd(&d_input[in_idx], d_output[out_idx]);
    }
}
// ====================================================================
//                          5. MSE LOSS
// ====================================================================
__global__ void mse_diff_kernel(float* pred, float* target, float* diff_sq, size_t size) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                float diff = pred[i] - target[i];
                diff_sq[i] = diff * diff;
    }
}
// Backward kernel for MSE: dL/d(pred) = 2 * (pred - target) / N
__global__ void mse_backward_kernel(float* pred, float* target, float* grad_out, size_t size) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                // The gradient is (2 * difference) / size. 
                grad_out[i] = 2.0f * (pred[i] - target[i]) / size; 
    }
}
float mse_loss_kernel(float* pred, float* target, size_t size) {
            float* diff_sq_d;
            checkCudaErrors(cudaMalloc((void**)&diff_sq_d, size * sizeof(float)));
            mse_diff_kernel<<<get_1d_dims(size), 256>>>(pred, target, diff_sq_d, size);
            checkCudaErrors(cudaGetLastError());
            // A more performant implementation would use CUB or a custom GPU reduction.
            // For simplicity and avoiding external libraries, we do a host sync and sum.
            float* diff_sq_h = (float*)malloc(size * sizeof(float));
            checkCudaErrors(cudaMemcpy(diff_sq_h, diff_sq_d, size * sizeof(float), cudaMemcpyDeviceToHost));
            
            double sum = 0.0; // Use double for accumulation to prevent precision issues
            for (size_t i = 0; i < size; ++i) {
                sum += diff_sq_h[i];
    }
    checkCudaErrors(cudaFree(diff_sq_d));
    free(diff_sq_h);
    
    return (float)(sum / size);
}
// ====================================================================
//                          6. OPTIMIZER
// ====================================================================
__global__ void update_weights_kernel(float* weights, float* d_weights, size_t size, float lr) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                weights[i] = weights[i] - lr * d_weights[i];
    }
}
// Utility for Xavier initialization
void init_random(std::vector<float>& vec, int fan_in, int fan_out) {
            std::random_device rd;
            std::mt19937 gen(rd());
            float limit = sqrt(6.0f / (fan_in + fan_out));
            std::uniform_real_distribution<float> d(-limit, limit);
            for (auto& x : vec) x = d(gen);
}
// Utility to save weights
void save_weights(const std::string& filename, const std::vector<float>& data) {
            std::ofstream file(filename, std::ios::binary);
            if (file.is_open()) {
                uint32_t size = data.size();
                file.write(reinterpret_cast<const char*>(&size), sizeof(size));
                file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
                file.close();
    } else {
                std::cerr << "Error saving: " << filename << "\n";
    }
}
// Helper to allocate and copy Host data to Device
void allocate_and_copy(float*& device_ptr, const std::vector<float>& host_data) {
            size_t size = host_data.size() * sizeof(float);
            checkCudaErrors(cudaMalloc((void**)&device_ptr, size));
            checkCudaErrors(cudaMemcpy(device_ptr, host_data.data(), size, cudaMemcpyHostToDevice));
}
// Helper for allocating device buffers (no initial copy needed)
void allocate_device_buffer(float*& device_ptr, size_t size_elements) {
            checkCudaErrors(cudaMalloc((void**)&device_ptr, size_elements * sizeof(float)));
}
int main() {
            // 1. CONFIG & DATA
            int BATCH = 32;
            int EPOCHS = 10;
            int MAX_IMAGES = 1024; // Limit number of images for quick testing
            float LR = 0.001f;
            std::string data_path = "../data/cifar-10-batches-bin";
            CIFAR10Dataset dataset(data_path);
            dataset.load_data();
            if (dataset.get_num_train() == 0) return 1;
            // --- HOST WEIGHTS AND BIASES ---
            // Encoder: 32x32x3 -> Conv1 -> 32x32x256 -> MaxPool -> 16x16x256
            // 16x16x256 -> Conv2 -> 16x16x128 -> MaxPool -> 8x8x128 (Latent)
            std::vector<float> h_w1(256*3*3*3);      init_random(h_w1, 3*3*3, 256*3*3);
            std::vector<float> h_b1(256, 0.0f);
            std::vector<float> h_w2(128*256*3*3);    init_random(h_w2, 256*3*3, 128*3*3);
            std::vector<float> h_b2(128, 0.0f);
            // Decoder: 8x8x128 -> Conv3 -> 8x8x128 (Conv on latent to extract features)
            // 8x8x128 -> Upsample -> 16x16x128 -> Conv4 -> 16x16x256
            // 16x16x256 -> Upsample -> 32x32x256 -> Conv5 -> 32x32x3
            std::vector<float> h_w3(128*128*3*3);    init_random(h_w3, 128*3*3, 128*3*3);
            std::vector<float> h_b3(128, 0.0f);
            std::vector<float> h_w4(256*128*3*3);    init_random(h_w4, 128*3*3, 256*3*3);
            std::vector<float> h_b4(256, 0.0f);
            std::vector<float> h_w5(3*256*3*3);      init_random(h_w5, 256*3*3, 3*3*3);
            std::vector<float> h_b5(3, 0.0f);
            // --- DEVICE POINTERS & SIZES ---
            float *d_w1, *d_b1, *d_dw1, *d_db1;
            float *d_w2, *d_b2, *d_dw2, *d_db2;
            float *d_w3, *d_b3, *d_dw3, *d_db3;
            float *d_w4, *d_b4, *d_dw4, *d_db4;
            float *d_w5, *d_b5, *d_dw5, *d_db5;
            float *d_input, *d_l1_out, *d_l1_pool, *d_l2_out, *d_latent;
            float *d_l3_out, *d_l3_up, *d_l4_out, *d_l4_up, *d_final_out;
            float *d_d_input, *d_d_l1_out, *d_d_l1_pool, *d_d_l2_out, *d_d_latent;
            float *d_d_l3_out, *d_d_l3_up, *d_d_l4_out, *d_d_l4_up, *d_d_final_out;
            size_t size_input   = (size_t)BATCH * 32 * 32 * 3;
            size_t size_l1_out  = (size_t)BATCH * 32 * 32 * 256;
            size_t size_l1_pool = (size_t)BATCH * 16 * 16 * 256;
            size_t size_l2_out  = (size_t)BATCH * 16 * 16 * 128;
            size_t size_latent  = (size_t)BATCH * 8 * 8 * 128;
            // Decoder output sizes
            // d_l3_out is size_latent
            size_t size_l3_up   = (size_t)BATCH * 16 * 16 * 128;
            size_t size_l4_out  = (size_t)BATCH * 16 * 16 * 256;
            size_t size_l4_up   = (size_t)BATCH * 32 * 32 * 256;
            // 2. ALLOCATE AND COPY MEMORY
            std::cout << "Allocating and copying initial weights to GPU...\n";
            allocate_and_copy(d_w1, h_w1); allocate_and_copy(d_b1, h_b1);
            allocate_and_copy(d_w2, h_w2); allocate_and_copy(d_b2, h_b2);
            allocate_and_copy(d_w3, h_w3); allocate_and_copy(d_b3, h_b3);
            allocate_and_copy(d_w4, h_w4); allocate_and_copy(d_b4, h_b4);
            allocate_and_copy(d_w5, h_w5); allocate_and_copy(d_b5, h_b5);
            allocate_device_buffer(d_dw1, h_w1.size()); allocate_device_buffer(d_db1, h_b1.size());
            allocate_device_buffer(d_dw2, h_w2.size()); allocate_device_buffer(d_db2, h_b2.size());
            allocate_device_buffer(d_dw3, h_w3.size()); allocate_device_buffer(d_db3, h_b3.size());
            allocate_device_buffer(d_dw4, h_w4.size()); allocate_device_buffer(d_db4, h_b4.size());
            allocate_device_buffer(d_dw5, h_w5.size()); allocate_device_buffer(d_db5, h_b5.size());
            
            allocate_device_buffer(d_input, size_input);
            allocate_device_buffer(d_l1_out, size_l1_out);
            allocate_device_buffer(d_l1_pool, size_l1_pool);
            allocate_device_buffer(d_l2_out, size_l2_out);
            allocate_device_buffer(d_latent, size_latent);
            allocate_device_buffer(d_l3_out, size_latent); 
            allocate_device_buffer(d_l3_up, size_l3_up); // Fixed size
            allocate_device_buffer(d_l4_out, size_l4_out); // Fixed size
            allocate_device_buffer(d_l4_up, size_l4_up); // Fixed size
            allocate_device_buffer(d_final_out, size_input);
            allocate_device_buffer(d_d_input, size_input);
            allocate_device_buffer(d_d_l1_out, size_l1_out);
            allocate_device_buffer(d_d_l1_pool, size_l1_pool);
            allocate_device_buffer(d_d_l2_out, size_l2_out);
            allocate_device_buffer(d_d_latent, size_latent);
            allocate_device_buffer(d_d_l3_out, size_latent);
            allocate_device_buffer(d_d_l3_up, size_l3_up); // Fixed size
            allocate_device_buffer(d_d_l4_out, size_l4_out); // Fixed size
            allocate_device_buffer(d_d_l4_up, size_l4_up); // Fixed size
            allocate_device_buffer(d_d_final_out, size_input);
            // 3. TRAINING LOOP
            std::cout << "--- START FULL TRAINING (CUDA) ---\n";
            
            // ConvParam_G: B, H_in, W_in, C_in, H_out, W_out, C_out, K, S, P
            // Encoder
            ConvParam_G p1 = {BATCH, 32, 32, 3,   32, 32, 256, 3, 1, 1}; // Output: 32x32x256
    ConvParam_G p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1}; // Output: 16x16x128
    // Decoder
    ConvParam_G p3 = {BATCH, 8, 8, 128,   8, 8, 128,   3, 1, 1}; // Output: 8x8x128 (Latent conv)
    ConvParam_G p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1}; // Output: 16x16x256
    ConvParam_G p5 = {BATCH, 32, 32, 256, 32, 32, 3,   3, 1, 1}; // Output: 32x32x3
    int num_batches = MAX_IMAGES / BATCH;
    auto start_total = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
                float total_loss = 0.0f;
                for (int b = 0; b < num_batches; ++b) {
                    // A. Copy Batch to Device
                    size_t offset = (size_t)b * (size_input / BATCH) * BATCH;
                    checkCudaErrors(cudaMemcpy(d_input, 
                                             dataset.get_train_images_ptr() + offset, 
                                             size_input * sizeof(float), 
                                             cudaMemcpyHostToDevice));
                    // B. FORWARD PASS (Direct Kernel Launches)
                    // Conv1 -> ReLU -> MaxPool (32->16)
                    conv2d_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_input, d_w1, d_b1, d_l1_out, p1);
                    checkCudaErrors(cudaGetLastError());
                    relu_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_l1_out, size_l1_out);
                    checkCudaErrors(cudaGetLastError());
                    maxpool_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_l1_out, d_l1_pool, BATCH, 32, 32, 256);
                    checkCudaErrors(cudaGetLastError());
                    // Conv2 -> ReLU -> MaxPool (16->8)
                    conv2d_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_l1_pool, d_w2, d_b2, d_l2_out, p2);
                    checkCudaErrors(cudaGetLastError());
                    relu_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_l2_out, size_l2_out);
                    checkCudaErrors(cudaGetLastError());
                    maxpool_kernel<<<get_1d_dims(size_latent), 256>>>(d_l2_out, d_latent, BATCH, 16, 16, 128);
                    checkCudaErrors(cudaGetLastError());
                    
                    // Conv3 (Latent) -> ReLU -> Upsample (8->16)
                    conv2d_kernel<<<get_1d_dims(size_latent), 256>>>(d_latent, d_w3, d_b3, d_l3_out, p3);
                    checkCudaErrors(cudaGetLastError());
                    relu_kernel<<<get_1d_dims(size_latent), 256>>>(d_l3_out, size_latent);
                    checkCudaErrors(cudaGetLastError());
                    upsample_kernel<<<get_1d_dims(size_l3_up), 256>>>(d_l3_out, d_l3_up, BATCH, 8, 8, 128); // 8x8 to 16x16
                    checkCudaErrors(cudaGetLastError());
                    // Conv4 -> ReLU -> Upsample (16->32)
                    conv2d_kernel<<<get_1d_dims(size_l4_out), 256>>>(d_l3_up, d_w4, d_b4, d_l4_out, p4);
                    checkCudaErrors(cudaGetLastError());
                    relu_kernel<<<get_1d_dims(size_l4_out), 256>>>(d_l4_out, size_l4_out);
                    checkCudaErrors(cudaGetLastError());
                    upsample_kernel<<<get_1d_dims(size_l4_up), 256>>>(d_l4_out, d_l4_up, BATCH, 16, 16, 256); // 16x16 to 32x32
                    checkCudaErrors(cudaGetLastError());
                    // Conv5 (Final output)
                    conv2d_kernel<<<get_1d_dims(size_input), 256>>>(d_l4_up, d_w5, d_b5, d_final_out, p5);
                    checkCudaErrors(cudaGetLastError());
                    // C. Loss (Assumed Host Wrapper with internal sync/copy)
                    float loss = mse_loss_kernel(d_final_out, d_input, size_input);
                    total_loss += loss;
                    
                    // D. BACKWARD PASS (Direct Kernel Launches)
                    
                    // 1. Final Output Gradient (MSE)
                    mse_backward_kernel<<<get_1d_dims(size_input), 256>>>(d_final_out, d_input, d_d_final_out, size_input);
                    checkCudaErrors(cudaGetLastError());
                    // 2. Conv5 Backward
                    // Zero out gradient buffers for accumulation
                    fill_zeros<<<get_1d_dims(h_w5.size()), 256>>>(d_dw5, h_w5.size());
                    fill_zeros<<<get_1d_dims(h_b5.size()), 256>>>(d_db5, h_b5.size());
                    fill_zeros<<<get_1d_dims(size_l4_up), 256>>>(d_d_l4_up, size_l4_up); // d_input buffer
                    conv2d_backward_input_kernel<<<get_1d_dims(size_l4_up), 256>>>(d_d_final_out, d_w5, d_d_l4_up, p5); // d_input
                    conv2d_backward_weight_kernel<<<get_1d_dims(h_w5.size()), 256>>>(d_d_final_out, d_l4_up, d_dw5, p5); // d_weight
                    conv2d_backward_bias_kernel<<<get_1d_dims(h_b5.size()), 256>>>(d_d_final_out, d_db5, p5); // d_bias
                    checkCudaErrors(cudaGetLastError());
                    
                    // 3. Upsample (32->16) Backward
                    fill_zeros<<<get_1d_dims(size_l4_out), 256>>>(d_d_l4_out, size_l4_out); // d_input buffer
                    upsample_backward_kernel<<<get_1d_dims(size_l4_up), 256>>>(d_d_l4_up, d_d_l4_out, BATCH, 16, 16, 256); // input_H=16, input_W=16
                    checkCudaErrors(cudaGetLastError());
                    // 4. Conv4 ReLU Backward
                    relu_backward_kernel<<<get_1d_dims(size_l4_out), 256>>>(d_d_l4_out, d_l4_out, d_d_l4_out, size_l4_out);
                    checkCudaErrors(cudaGetLastError());
                    // 5. Conv4 Backward
                    fill_zeros<<<get_1d_dims(h_w4.size()), 256>>>(d_dw4, h_w4.size());
                    fill_zeros<<<get_1d_dims(h_b4.size()), 256>>>(d_db4, h_b4.size());
                    fill_zeros<<<get_1d_dims(size_l3_up), 256>>>(d_d_l3_up, size_l3_up); // d_input buffer
                    conv2d_backward_input_kernel<<<get_1d_dims(size_l3_up), 256>>>(d_d_l4_out, d_w4, d_d_l3_up, p4);
                    conv2d_backward_weight_kernel<<<get_1d_dims(h_w4.size()), 256>>>(d_d_l4_out, d_l3_up, d_dw4, p4);
                    conv2d_backward_bias_kernel<<<get_1d_dims(h_b4.size()), 256>>>(d_d_l4_out, d_db4, p4);
                    checkCudaErrors(cudaGetLastError());
                    // 6. Upsample (16->8) Backward
                    fill_zeros<<<get_1d_dims(size_latent), 256>>>(d_d_l3_out, size_latent); // d_input buffer
                    upsample_backward_kernel<<<get_1d_dims(size_l3_up), 256>>>(d_d_l3_up, d_d_l3_out, BATCH, 8, 8, 128); // input_H=8, input_W=8
                    checkCudaErrors(cudaGetLastError());
                    // 7. Conv3 ReLU Backward
                    relu_backward_kernel<<<get_1d_dims(size_latent), 256>>>(d_d_l3_out, d_l3_out, d_d_l3_out, size_latent);
                    checkCudaErrors(cudaGetLastError());
                    // 8. Conv3 Backward (Latent)
                    fill_zeros<<<get_1d_dims(h_w3.size()), 256>>>(d_dw3, h_w3.size());
                    fill_zeros<<<get_1d_dims(h_b3.size()), 256>>>(d_db3, h_b3.size());
                    fill_zeros<<<get_1d_dims(size_latent), 256>>>(d_d_latent, size_latent); // d_input buffer
                    conv2d_backward_input_kernel<<<get_1d_dims(size_latent), 256>>>(d_d_l3_out, d_w3, d_d_latent, p3);
                    conv2d_backward_weight_kernel<<<get_1d_dims(h_w3.size()), 256>>>(d_d_l3_out, d_latent, d_dw3, p3);
                    conv2d_backward_bias_kernel<<<get_1d_dims(h_b3.size()), 256>>>(d_d_l3_out, d_db3, p3);
                    checkCudaErrors(cudaGetLastError());
                    // 9. MaxPool (16->8) Backward
                    fill_zeros<<<get_1d_dims(size_l2_out), 256>>>(d_d_l2_out, size_l2_out); // d_input buffer
                    maxpool_backward_kernel<<<get_1d_dims(size_latent), 256>>>(d_d_latent, d_l2_out, d_d_l2_out, BATCH, 16, 16, 128); // input_H=16, input_W=16
                    checkCudaErrors(cudaGetLastError());
                    // 10. Conv2 ReLU Backward
                    relu_backward_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_d_l2_out, d_l2_out, d_d_l2_out, size_l2_out);
                    checkCudaErrors(cudaGetLastError());
                    // 11. Conv2 Backward
                    fill_zeros<<<get_1d_dims(h_w2.size()), 256>>>(d_dw2, h_w2.size());
                    fill_zeros<<<get_1d_dims(h_b2.size()), 256>>>(d_db2, h_b2.size());
                    fill_zeros<<<get_1d_dims(size_l1_pool), 256>>>(d_d_l1_pool, size_l1_pool); // d_input buffer
                    conv2d_backward_input_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_d_l2_out, d_w2, d_d_l1_pool, p2);
                    conv2d_backward_weight_kernel<<<get_1d_dims(h_w2.size()), 256>>>(d_d_l2_out, d_l1_pool, d_dw2, p2);
                    conv2d_backward_bias_kernel<<<get_1d_dims(h_b2.size()), 256>>>(d_d_l2_out, d_db2, p2);
                    checkCudaErrors(cudaGetLastError());
                    // 12. MaxPool (32->16) Backward
                    fill_zeros<<<get_1d_dims(size_l1_out), 256>>>(d_d_l1_out, size_l1_out); // d_input buffer
                    maxpool_backward_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_d_l1_pool, d_l1_out, d_d_l1_out, BATCH, 32, 32, 256); // input_H=32, input_W=32
                    checkCudaErrors(cudaGetLastError());
                    // 13. Conv1 ReLU Backward
                    relu_backward_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_d_l1_out, d_l1_out, d_d_l1_out, size_l1_out);
                    checkCudaErrors(cudaGetLastError());
                    // 14. Conv1 Backward
                    fill_zeros<<<get_1d_dims(h_w1.size()), 256>>>(d_dw1, h_w1.size());
                    fill_zeros<<<get_1d_dims(h_b1.size()), 256>>>(d_db1, h_b1.size());
                    fill_zeros<<<get_1d_dims(size_input), 256>>>(d_d_input, size_input); // d_input buffer (optional, likely discarded)
                    conv2d_backward_input_kernel<<<get_1d_dims(size_input), 256>>>(d_d_l1_out, d_w1, d_d_input, p1);
                    conv2d_backward_weight_kernel<<<get_1d_dims(h_w1.size()), 256>>>(d_d_l1_out, d_input, d_dw1, p1);
                    conv2d_backward_bias_kernel<<<get_1d_dims(h_b1.size()), 256>>>(d_d_l1_out, d_db1, p1);
                    checkCudaErrors(cudaGetLastError());
                    // E. Update Weights
                    update_weights_kernel<<<get_1d_dims(h_w1.size()), 256>>>(d_w1, d_dw1, h_w1.size(), LR); 
                    update_weights_kernel<<<get_1d_dims(h_b1.size()), 256>>>(d_b1, d_db1, h_b1.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_w2.size()), 256>>>(d_w2, d_dw2, h_w2.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_b2.size()), 256>>>(d_b2, d_db2, h_b2.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_w3.size()), 256>>>(d_w3, d_dw3, h_w3.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_b3.size()), 256>>>(d_b3, d_db3, h_b3.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_w4.size()), 256>>>(d_w4, d_dw4, h_w4.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_b4.size()), 256>>>(d_b4, d_db4, h_b4.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_w5.size()), 256>>>(d_w5, d_dw5, h_w5.size(), LR);
                    update_weights_kernel<<<get_1d_dims(h_b5.size()), 256>>>(d_b5, d_db5, h_b5.size(), LR);
                    checkCudaErrors(cudaGetLastError());
}
    auto end_epoch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_epoch = end_epoch - start_total;
    std::cout << "\nEpoch " << epoch + 1 << " Done. Avg Loss: " << total_loss / num_batches 
                << " | Time: " << elapsed_epoch.count() << "s\n";
    } // End of epoch loop
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "\n--- Training Complete ---\n";
    std::cout << "Total Training Time: " << elapsed_total.count() << " seconds\n";
    
    // 4. COPY FINAL WEIGHTS BACK TO HOST & SAVE
    std::cout << "\n--- Copying Final Weights to Host and Saving ---\n";
    // Copy Weights back D -> H
    checkCudaErrors(cudaMemcpy(h_w1.data(), d_w1, h_w1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_b1.data(), d_b1, h_b1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_w2.data(), d_w2, h_w2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_b2.data(), d_b2, h_b2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_w3.data(), d_w3, h_w3.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_b3.data(), d_b3, h_b3.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_w4.data(), d_w4, h_w4.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_b4.data(), d_b4, h_b4.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_w5.data(), d_w5, h_w5.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_b5.data(), d_b5, h_b5.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // Save Weights (using the host save_weights utility function)
    save_weights("../weights/enc_w1.bin", h_w1); save_weights("../weights/enc_b1.bin", h_b1);
    save_weights("../weights/enc_w2.bin", h_w2); save_weights("../weights/enc_b2.bin", h_b2);
    save_weights("../weights/dec_w3.bin", h_w3); save_weights("../weights/dec_b3.bin", h_b3);
    save_weights("../weights/dec_w4.bin", h_w4); save_weights("../weights/dec_b4.bin", h_b4);
    save_weights("../weights/dec_w5.bin", h_w5); save_weights("../weights/dec_b5.bin", h_b5);
    // 5. CLEANUP DEVICE MEMORY
    std::cout << "\n--- Cleaning up device memory ---\n";
    
    // Free Weights and Gradients
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_dw1); cudaFree(d_db1);
    cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_dw2); cudaFree(d_db2);
    cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_dw3); cudaFree(d_db3);
    cudaFree(d_w4); cudaFree(d_b4); cudaFree(d_dw4); cudaFree(d_db4);
    cudaFree(d_w5); cudaFree(d_b5); cudaFree(d_dw5); cudaFree(d_db5);
    
    // Free Forward Buffers
    cudaFree(d_input); cudaFree(d_l1_out); cudaFree(d_l1_pool); cudaFree(d_l2_out); cudaFree(d_latent);
    cudaFree(d_l3_out); cudaFree(d_l3_up); cudaFree(d_l4_out); cudaFree(d_l4_up); cudaFree(d_final_out);
    
    // Free Backward Buffers
    cudaFree(d_d_input); cudaFree(d_d_l1_out); cudaFree(d_d_l1_pool); cudaFree(d_d_l2_out); cudaFree(d_d_latent);
    cudaFree(d_d_l3_out); cudaFree(d_d_l3_up); cudaFree(d_d_l4_out); cudaFree(d_d_l4_up); cudaFree(d_d_final_out);
    std::cout << "Cleanup complete. Exiting program.\n";
    return 0;
} // End of main function