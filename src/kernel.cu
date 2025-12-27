#include "kernel.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

// CUDA ERROR CHECKING
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, line, 
                static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

void checkCudaErrors(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " (Code: " << code << ")\n";
        exit(code);
    }
}

// KERNEL LAUNCH CONFIGURATION
dim3 get_1d_dims(size_t total_size) {
    const int THREADS_PER_BLOCK = 256;
    size_t blocks = (total_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return dim3((unsigned int)blocks, 1, 1);
}

// DEVICE HELPER FUNCTIONS
__device__ inline int get_idx_dev(int b, int h, int w, int c, int H, int W, int C) {
    return b * (H * W * C) + h * (W * C) + w * C + c;
}

__global__ void fill_zeros(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 0.0f;
}

// CONVOLUTION KERNELS
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

        for (int ic = 0; ic < p.C_in; ++ic) {
            for (int kh = 0; kh < p.K; ++kh) {
                for (int kw = 0; kw < p.K; ++kw) {
                    int ih = oh * p.S - p.P + kh;
                    int iw = ow * p.S - p.P + kw;

                    if (ih >= 0 && ih < p.H_in && iw >= 0 && iw < p.W_in) {
                        int in_idx = get_idx_dev(b, ih, iw, ic, p.H_in, p.W_in, p.C_in);
                        int w_idx = oc * (p.C_in * p.K * p.K) + ic * (p.K * p.K) + kh * p.K + kw;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[out_idx] = sum;
    }
}

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

        for (int oc = 0; oc < p.C_out; ++oc) {
            for (int kh = 0; kh < p.K; ++kh) {
                for (int kw = 0; kw < p.K; ++kw) {
                    int h_shifted = h + p.P - kh;
                    int w_shifted = w + p.P - kw;

                    if (h_shifted % p.S == 0 && w_shifted % p.S == 0) {
                        int oh = h_shifted / p.S;
                        int ow = w_shifted / p.S;

                        if (oh >= 0 && oh < p.H_out && ow >= 0 && ow < p.W_out) {
                            int out_idx = get_idx_dev(b, oh, ow, oc, p.H_out, p.W_out, p.C_out);
                            int w_idx = oc * (p.C_in * p.K * p.K) + c * (p.K * p.K) + kh * p.K + kw;
                            sum += d_output[out_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        d_input[idx] = sum;
    }
}

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

// RELU KERNELS
__global__ void relu_kernel(float* data, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = (data[i] < 0.0f) ? 0.0f : data[i];
    }
}

__global__ void relu_backward_kernel(float* d_output, float* input, float* d_input, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_input[i] = (input[i] > 0) ? d_output[i] : 0.0f;
    }
}

// MAX POOLING KERNELS
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

        if (max_idx != -1) {
            atomicAdd(&d_input[max_idx], d_output[out_idx]);
        }
    }
}

// UPSAMPLE KERNELS
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

__global__ void upsample_backward_kernel(float* d_output, float* d_input, 
                                         int batch, int in_h, int in_w, int in_c) {
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

        atomicAdd(&d_input[in_idx], d_output[out_idx]);
    }
}

// MSE LOSS KERNELS
__global__ void mse_diff_kernel(float* pred, float* target, float* diff_sq, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float diff = pred[i] - target[i];
        diff_sq[i] = diff * diff;
    }
}

__global__ void mse_backward_kernel(float* pred, float* target, float* grad_out, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_out[i] = 2.0f * (pred[i] - target[i]) / size;
    }
}

float mse_loss_kernel(float* pred, float* target, size_t size) {
    float* diff_sq_d;
    checkCudaErrors(cudaMalloc((void**)&diff_sq_d, size * sizeof(float)));

    mse_diff_kernel<<<get_1d_dims(size), 256>>>(pred, target, diff_sq_d, size);
    checkCudaErrors(cudaGetLastError());
    
    float* diff_sq_h = (float*)malloc(size * sizeof(float));
    checkCudaErrors(cudaMemcpy(diff_sq_h, diff_sq_d, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += diff_sq_h[i];
    }

    checkCudaErrors(cudaFree(diff_sq_d));
    free(diff_sq_h);
    
    return (float)(sum / size);
}

// OPTIMIZER KERNELS
__global__ void update_weights_kernel(float* weights, float* d_weights, size_t size, float lr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        weights[i] = weights[i] - lr * d_weights[i];
    }
}

// UTILITY FUNCTIONS
void init_random(std::vector<float>& vec, int fan_in, int fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> d(-limit, limit);
    for (auto& x : vec) x = d(gen);
}

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

bool load_weights(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    return true;
}

// MEMORY ALLOCATION HELPERS
void allocate_and_copy(float*& device_ptr, const std::vector<float>& host_data) {
    size_t size = host_data.size() * sizeof(float);
    checkCudaErrors(cudaMalloc((void**)&device_ptr, size));
    checkCudaErrors(cudaMemcpy(device_ptr, host_data.data(), size, cudaMemcpyHostToDevice));
}

void allocate_device_buffer(float*& device_ptr, size_t size_elements) {
    checkCudaErrors(cudaMalloc((void**)&device_ptr, size_elements * sizeof(float)));
}