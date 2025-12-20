#ifndef KERNEL_CPU_H
#define KERNEL_CPU_H

#include <vector>
#include <string>

// ============== STRUCTURES ==============
struct ConvParam {
    int batch, in_c, in_h, in_w;
    int out_c, out_h, out_w;
    int k_size, stride, padding;
};

// ============== CONVOLUTION FUNCTIONS ==============
void conv2d(float* input, float* weight, float* bias, float* output, ConvParam p);
void conv2d_backward(float* d_output, float* input, float* weight, 
                     float* d_input, float* d_weight, float* d_bias, ConvParam p);

// ============== RELU FUNCTIONS ==============
void relu(float* data, int size);
void relu_backward(float* d_output, float* input, float* d_input, int size);

// ============== MAX POOLING FUNCTIONS ==============
void maxpool(float* input, float* output, int batch, int in_h, int in_w, int in_c);
void maxpool_backward(float* d_output, float* input, float* d_input, 
                      int batch, int in_h, int in_w, int in_c);

// ============== UPSAMPLE FUNCTIONS ==============
void upsample(float* input, float* output, int batch, int in_h, int in_w, int in_c);
void upsample_backward(float* d_output, float* d_input, int batch, int in_h, int in_w, int in_c);

// ============== MSE LOSS FUNCTIONS ==============
float mse_loss(float* pred, float* target, int size);

// ============== OPTIMIZER FUNCTIONS ==============
void update_weights(float* weights, float* d_weights, int size, float lr);

// ============== UTILITY FUNCTIONS ==============
void init_random(std::vector<float>& vec, int fan_in, int fan_out);
void save_weights(const std::string& filename, const std::vector<float>& data);
bool load_weights(const std::string& filename, std::vector<float>& data);

#endif // KERNEL_CPU_H
