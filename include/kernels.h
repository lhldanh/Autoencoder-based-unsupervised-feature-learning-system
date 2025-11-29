#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <cmath>
#include <iostream>

struct ConvParam {
    int batch;
    int in_h, in_w, in_c;
    int out_h, out_w, out_c;
    int k_size;
    int stride;
    int padding;
};

// --- CPU DECLARATIONS ---
void conv2d_cpu(float* input, float* weight, float* bias, float* output, ConvParam p);
void relu_cpu(float* data, int size);
void maxpool_cpu(float* input, float* output, int batch, int in_h, int in_w, int in_c);
void upsample_cpu(float* input, float* output, int batch, int in_h, int in_w, int in_c);
float mse_loss_cpu(float* pred, float* target, int size);
void conv2d_backward_cpu(float* d_output, float* input, float* weight, 
                         float* d_input, float* d_weight, float* d_bias, ConvParam p);
void relu_backward_cpu(float* d_output, float* input, float* d_input, int size);
void maxpool_backward_cpu(float* d_output, float* input, float* d_input, 
                          int batch, int in_h, int in_w, int in_c);
void update_weights_cpu(float* weights, float* d_weights, int size, float lr);
void upsample_backward_cpu(float* d_output, float* d_input, int batch, int in_h, int in_w, int in_c);
#endif