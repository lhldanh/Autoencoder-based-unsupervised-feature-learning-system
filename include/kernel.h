#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>

// ============== STRUCTURES ==============
struct ConvParam_G {
    int B, H_in, W_in, C_in;
    int H_out, W_out, C_out;
    int K, S, P;
};

// ============== CUDA ERROR CHECKING ==============
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
void checkCudaErrors(cudaError_t code);

#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)

// ============== KERNEL LAUNCH CONFIGURATION ==============
dim3 get_1d_dims(size_t total_size);

// ============== DEVICE HELPER KERNELS ==============
__global__ void fill_zeros(float* data, size_t size);

// ============== CONVOLUTION KERNELS ==============
__global__ void conv2d_kernel(float* input, float* weight, float* bias, float* output, ConvParam_G p);
__global__ void conv2d_backward_input_kernel(float* d_output, float* weight, float* d_input, ConvParam_G p);
__global__ void conv2d_backward_weight_kernel(float* d_output, float* input, float* d_weight, ConvParam_G p);
__global__ void conv2d_backward_bias_kernel(float* d_output, float* d_bias, ConvParam_G p);

// ============== RELU KERNELS ==============
__global__ void relu_kernel(float* data, size_t size);
__global__ void relu_backward_kernel(float* d_output, float* input, float* d_input, size_t size);

// ============== MAX POOLING KERNELS ==============
__global__ void maxpool_kernel(float* input, float* output, int batch, int in_h, int in_w, int in_c);
__global__ void maxpool_backward_kernel(float* d_output, float* input, float* d_input, 
                                        int batch, int in_h, int in_w, int in_c);

// ============== UPSAMPLE KERNELS ==============
__global__ void upsample_kernel(float* input, float* output, int batch, int in_h, int in_w, int in_c);
__global__ void upsample_backward_kernel(float* d_output, float* d_input, 
                                         int batch, int in_h, int in_w, int in_c);

// ============== MSE LOSS KERNELS ==============
__global__ void mse_diff_kernel(float* pred, float* target, float* diff_sq, size_t size);
__global__ void mse_backward_kernel(float* pred, float* target, float* grad_out, size_t size);
float mse_loss_kernel(float* pred, float* target, size_t size);

// ============== OPTIMIZER KERNELS ==============
__global__ void update_weights_kernel(float* weights, float* d_weights, size_t size, float lr);

// ============== UTILITY FUNCTIONS ==============
void init_random(std::vector<float>& vec, int fan_in, int fan_out);
void save_weights(const std::string& filename, const std::vector<float>& data);
bool load_weights(const std::string& filename, std::vector<float>& data);

// ============== MEMORY ALLOCATION HELPERS ==============
void allocate_and_copy(float*& device_ptr, const std::vector<float>& host_data);
void allocate_device_buffer(float*& device_ptr, size_t size_elements);

#endif // KERNEL_H