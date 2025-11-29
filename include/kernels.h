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

// --- FORWARD DECLARATIONS (GENERIC NAMES) ---
void conv2d(float* input, float* weight, float* bias, float* output, ConvParam p);
void relu(float* data, int size);
void maxpool(float* input, float* output, int batch, int in_h, int in_w, int in_c);
void upsample(float* input, float* output, int batch, int in_h, int in_w, int in_c);
float mse_loss(float* pred, float* target, int size);

// --- BACKWARD DECLARATIONS (GENERIC NAMES) ---
void conv2d_backward(float* d_output, float* input, float* weight, 
                     float* d_input, float* d_weight, float* d_bias, ConvParam p);
void relu_backward(float* d_output, float* input, float* d_input, int size);
void maxpool_backward(float* d_output, float* input, float* d_input, 
                      int batch, int in_h, int in_w, int in_c);
void upsample_backward(float* d_output, float* d_input, int batch, int in_h, int in_w, int in_c);
void update_weights(float* weights, float* d_weights, int size, float lr);

#endif

#pragma once

#include <cmath>
#include <cstddef> // for size_t

// Structure for Convolution Parameters
struct ConvParam_G {
    int B, H_in, W_in, C_in; // Batch, Height In, Width In, Channels In
    int H_out, W_out, C_out; // Height Out, Width Out, Channels Out
    int K, S, P;             // Kernel size, Stride, Padding
};

// --- CPU (Host) Prototypes (Implemented in a separate file, e.g., kernels_host.cpp) ---

void conv2d_cpu(const float* input, const float* weights, const float* bias, float* output, const ConvParam_G& p);
void relu_cpu(float* data, size_t size);
void maxpool_cpu(const float* input, float* output, int B, int H_in, int W_in, int C);
void upsample_cpu(const float* input, float* output, int B, int H_in, int W_in, int C);

float mse_loss_cpu(const float* output, const float* target, size_t size);

void conv2d_backward_cpu(const float* d_output, const float* input, const float* weights,
                         float* d_input, float* d_weights, float* d_bias, const ConvParam_G& p);
void relu_backward_cpu(const float* d_output, const float* output, float* d_input, size_t size);
void maxpool_backward_cpu(const float* d_output, const float* input, float* d_input, 
                          int B, int H_in, int W_in, int C);
void upsample_backward_cpu(const float* d_output, float* d_input, int B, int H_in, int W_in, int C);

void update_weights_cpu(float* weights, const float* gradients, size_t size, float lr);

// --- GPU (Device) Prototypes (Implemented in a separate file, e.g., kernels.cu) ---

// Kernels are typically wrapped in host functions for launch configuration.
// These functions will handle cudaMalloc/free only for temporary buffers and kernel launch.
void conv2d_gpu(const float* d_input, const float* d_weights, const float* d_bias, float* d_output, const ConvParam_G& p);
void relu_gpu(float* d_data, size_t size);
void maxpool_gpu(const float* d_input, float* d_output, int B, int H_in, int W_in, int C);
void upsample_gpu(const float* d_input, float* d_output, int B, int H_in, int W_in, int C);

float mse_loss_gpu(const float* d_output, const float* d_target, size_t size);

void conv2d_backward_gpu(const float* d_d_output, const float* d_input, const float* d_weights,
                         float* d_d_input, float* d_d_weights, float* d_d_bias, const ConvParam_G& p);
void relu_backward_gpu(const float* d_d_output, const float* d_output, float* d_d_input, size_t size);
void maxpool_backward_gpu(const float* d_d_output, const float* d_input, float* d_d_input, 
                          int B, int H_in, int W_in, int C);
void upsample_backward_gpu(const float* d_d_output, float* d_d_input, int B, int H_in, int W_in, int C);

void update_weights_gpu(float* d_weights, const float* d_gradients, size_t size, float lr);

void conv2d_backward_gpu(const float* d_d_output, const float* d_input, const float* d_weights,
                         float* d_d_input, float* d_d_weights, float* d_d_bias, const ConvParam_G& p);