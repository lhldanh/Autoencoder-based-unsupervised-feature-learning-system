#include <iostream>
#include <vector>
#include <chrono>
#include "cifar10_dataset.h"
#include "optimize_kernel.h"
#include "kernel.h"

int main() {
    const int B = 64, EPOCHS = 5;
    const int max_images = 96;
    
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin");
    dataset.load_data();
    if (dataset.get_num_train() == 0) return 1;
    
    int s_in = B * 32 * 32 * 3;
    int s_l1 = B * 32 * 32 * 256, s_p1 = B * 16 * 16 * 256;
    int s_l2 = B * 16 * 16 * 128, s_p2 = B * 8 * 8 * 128;
    int s_l3 = B * 8 * 8 * 128,   s_u3 = B * 16 * 16 * 128;
    int s_l4 = B * 16 * 16 * 256, s_u4 = B * 32 * 32 * 256;
    
    std::vector<float> h_w1(256 * 3 * 9), h_b1(256, 0);
    std::vector<float> h_w2(128 * 256 * 9), h_b2(128, 0);
    std::vector<float> h_w3(128 * 128 * 9), h_b3(128, 0);
    std::vector<float> h_w4(256 * 128 * 9), h_b4(256, 0);
    std::vector<float> h_w5(3 * 256 * 9), h_b5(3, 0);
    
    init_random(h_w1, 27, 256); init_random(h_w2, 2304, 128);
    init_random(h_w3, 1152, 128); init_random(h_w4, 1152, 256);
    init_random(h_w5, 2304, 3);
    init_random(h_b1, 27, 256);
    init_random(h_b2, 2304, 128);
    init_random(h_b3, 1152, 128);
    init_random(h_b4, 1152, 256);
    init_random(h_b5, 2304, 3);
    
    float *d_w1, *d_b1, *d_dw1, *d_db1;
    float *d_w2, *d_b2, *d_dw2, *d_db2;
    float *d_w3, *d_b3, *d_dw3, *d_db3;
    float *d_w4, *d_b4, *d_dw4, *d_db4;
    float *d_w5, *d_b5, *d_dw5, *d_db5;
    
    cudaMalloc(&d_w1, h_w1.size() * 4); cudaMalloc(&d_b1, 256 * 4);
    cudaMalloc(&d_dw1, h_w1.size() * 4); cudaMalloc(&d_db1, 256 * 4);
    cudaMalloc(&d_w2, h_w2.size() * 4); cudaMalloc(&d_b2, 128 * 4);
    cudaMalloc(&d_dw2, h_w2.size() * 4); cudaMalloc(&d_db2, 128 * 4);
    cudaMalloc(&d_w3, h_w3.size() * 4); cudaMalloc(&d_b3, 128 * 4);
    cudaMalloc(&d_dw3, h_w3.size() * 4); cudaMalloc(&d_db3, 128 * 4);
    cudaMalloc(&d_w4, h_w4.size() * 4); cudaMalloc(&d_b4, 256 * 4);
    cudaMalloc(&d_dw4, h_w4.size() * 4); cudaMalloc(&d_db4, 256 * 4);
    cudaMalloc(&d_w5, h_w5.size() * 4); cudaMalloc(&d_b5, 3 * 4);
    cudaMalloc(&d_dw5, h_w5.size() * 4); cudaMalloc(&d_db5, 3 * 4);
    
    float *d_input[2];
    cudaMalloc(&d_input[0], s_in * 4);
    cudaMalloc(&d_input[1], s_in * 4);
    
    float *d_l1, *d_p1, *d_l2, *d_p2, *d_l3, *d_u3, *d_l4, *d_u4, *d_out;
    cudaMalloc(&d_l1, s_l1 * 4); cudaMalloc(&d_p1, s_p1 * 4);
    cudaMalloc(&d_l2, s_l2 * 4); cudaMalloc(&d_p2, s_p2 * 4);
    cudaMalloc(&d_l3, s_l3 * 4); cudaMalloc(&d_u3, s_u3 * 4);
    cudaMalloc(&d_l4, s_l4 * 4); cudaMalloc(&d_u4, s_u4 * 4);
    cudaMalloc(&d_out, s_in * 4);
    
    float* d_loss;
    cudaMalloc(&d_loss, 4);
    
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1.data(), 256 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2.data(), 128 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3.data(), h_w3.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3.data(), 128 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4, h_w4.data(), h_w4.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4, h_b4.data(), 256 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5, h_w5.data(), h_w5.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5, h_b5.data(), 3 * 4, cudaMemcpyHostToDevice);
        ConvParam_G p1 = {B, 32, 32, 3,  32, 32, 256, 3, 1, 1};
        ConvParam_G p2 = {B, 16, 16, 256, 16, 16, 128, 3, 1, 1};
        ConvParam_G p3 = {B, 8, 8, 128,  8, 8, 128,  3, 1, 1};
        ConvParam_G p4 = {B, 16, 16, 128, 16, 16, 256, 3, 1, 1};
        ConvParam_G p5 = {B, 32, 32, 256, 32, 32, 3,  3, 1, 1};
        float LR = 0.001f;
    
        float *d_dl1, *d_dp1, *d_dl2, *d_dp2, *d_dl3, *d_du3, *d_dl4, *d_du4, *d_dout;
        cudaMalloc(&d_dl1, s_l1 * 4); cudaMalloc(&d_dp1, s_p1 * 4);
        cudaMalloc(&d_dl2, s_l2 * 4); cudaMalloc(&d_dp2, s_p2 * 4);
        cudaMalloc(&d_dl3, s_l3 * 4); cudaMalloc(&d_du3, s_u3 * 4);
        cudaMalloc(&d_dl4, s_l4 * 4); cudaMalloc(&d_du4, s_u4 * 4);
        cudaMalloc(&d_dout, s_in * 4);
    
    
    // Create CUDA streams for overlapping computation and data transfer
    cudaStream_t stream_compute, stream_transfer;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_transfer);
    
    int num_batches = max_images / B;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int b = 0; b < num_batches; ++b) {
            int buf_idx = b % 2;
            int next_buf = (b + 1) % 2;
            size_t offset = (size_t)b * s_in;
            
            // Transfer current batch
            cudaMemcpyAsync(d_input[buf_idx], dataset.get_train_images_ptr() + offset, 
                        s_in * sizeof(float), cudaMemcpyHostToDevice, stream_transfer);
            
            // Wait for transfer to complete before computation
            cudaStreamSynchronize(stream_transfer);
            
            // Pre-fetch next batch if available
            if (b + 1 < num_batches) {
                size_t next_offset = (size_t)(b + 1) * s_in;
                cudaMemcpyAsync(d_input[next_buf], dataset.get_train_images_ptr() + next_offset, 
                               s_in * sizeof(float), cudaMemcpyHostToDevice, stream_transfer);
            }

            // FORWARD on compute stream
            conv2d_kernel<<<get_1d_dims(s_l1), 256, 0, stream_compute>>>(d_input[buf_idx], d_w1, d_b1, d_l1, p1);
            relu_kernel<<<get_1d_dims(s_l1), 256, 0, stream_compute>>>(d_l1, s_l1);
            maxpool_kernel<<<get_1d_dims(s_p1), 256, 0, stream_compute>>>(d_l1, d_p1, B, 32, 32, 256);

            conv2d_kernel<<<get_1d_dims(s_l2), 256, 0, stream_compute>>>(d_p1, d_w2, d_b2, d_l2, p2);
            relu_kernel<<<get_1d_dims(s_l2), 256, 0, stream_compute>>>(d_l2, s_l2);
            maxpool_kernel<<<get_1d_dims(s_p2), 256, 0, stream_compute>>>(d_l2, d_p2, B, 16, 16, 128);
            
            conv2d_kernel<<<get_1d_dims(s_l3), 256, 0, stream_compute>>>(d_p2, d_w3, d_b3, d_l3, p3);
            relu_kernel<<<get_1d_dims(s_l3), 256, 0, stream_compute>>>(d_l3, s_l3);
            upsample_kernel<<<get_1d_dims(s_u3), 256, 0, stream_compute>>>(d_l3, d_u3, B, 8, 8, 128);

            conv2d_kernel<<<get_1d_dims(s_l4), 256, 0, stream_compute>>>(d_u3, d_w4, d_b4, d_l4, p4);
            relu_kernel<<<get_1d_dims(s_l4), 256, 0, stream_compute>>>(d_l4, s_l4);
            upsample_kernel<<<get_1d_dims(s_u4), 256, 0, stream_compute>>>(d_l4, d_u4, B, 16, 16, 256);

            conv2d_kernel<<<get_1d_dims(s_in), 256, 0, stream_compute>>>(d_u4, d_w5, d_b5, d_out, p5);

            // BACKWARD
            mse_backward_kernel<<<get_1d_dims(s_in), 256, 0, stream_compute>>>(d_out, d_input[buf_idx], d_dout, s_in);

            fill_zeros<<<get_1d_dims(h_w5.size()), 256, 0, stream_compute>>>(d_dw5, h_w5.size());
            fill_zeros<<<get_1d_dims(3), 256, 0, stream_compute>>>(d_db5, 3);
            fill_zeros<<<get_1d_dims(s_u4), 256, 0, stream_compute>>>(d_du4, s_u4);
            conv2d_backward_input_kernel<<<get_1d_dims(s_u4), 256, 0, stream_compute>>>(d_dout, d_w5, d_du4, p5);
            conv2d_backward_weight_kernel<<<get_1d_dims(h_w5.size()), 256, 0, stream_compute>>>(d_dout, d_u4, d_dw5, p5);
            conv2d_backward_bias_kernel<<<get_1d_dims(3), 256, 0, stream_compute>>>(d_dout, d_db5, p5);
            
            fill_zeros<<<get_1d_dims(s_l4), 256, 0, stream_compute>>>(d_dl4, s_l4);
            upsample_backward_kernel<<<get_1d_dims(s_u4), 256, 0, stream_compute>>>(d_du4, d_dl4, B, 16, 16, 256);
            relu_backward_kernel<<<get_1d_dims(s_l4), 256, 0, stream_compute>>>(d_dl4, d_l4, d_dl4, s_l4);

            fill_zeros<<<get_1d_dims(h_w4.size()), 256, 0, stream_compute>>>(d_dw4, h_w4.size());
            fill_zeros<<<get_1d_dims(256), 256, 0, stream_compute>>>(d_db4, 256);
            fill_zeros<<<get_1d_dims(s_u3), 256, 0, stream_compute>>>(d_du3, s_u3);
            conv2d_backward_input_kernel<<<get_1d_dims(s_u3), 256, 0, stream_compute>>>(d_dl4, d_w4, d_du3, p4);
            conv2d_backward_weight_kernel<<<get_1d_dims(h_w4.size()), 256, 0, stream_compute>>>(d_dl4, d_u3, d_dw4, p4);
            conv2d_backward_bias_kernel<<<get_1d_dims(256), 256, 0, stream_compute>>>(d_dl4, d_db4, p4);

            fill_zeros<<<get_1d_dims(s_l3), 256, 0, stream_compute>>>(d_dl3, s_l3);
            upsample_backward_kernel<<<get_1d_dims(s_u3), 256, 0, stream_compute>>>(d_du3, d_dl3, B, 8, 8, 128);
            relu_backward_kernel<<<get_1d_dims(s_l3), 256, 0, stream_compute>>>(d_dl3, d_l3, d_dl3, s_l3);

            fill_zeros<<<get_1d_dims(h_w3.size()), 256, 0, stream_compute>>>(d_dw3, h_w3.size());
            fill_zeros<<<get_1d_dims(128), 256, 0, stream_compute>>>(d_db3, 128);
            fill_zeros<<<get_1d_dims(s_p2), 256, 0, stream_compute>>>(d_dp2, s_p2);
            conv2d_backward_input_kernel<<<get_1d_dims(s_p2), 256, 0, stream_compute>>>(d_dl3, d_w3, d_dp2, p3);
            conv2d_backward_weight_kernel<<<get_1d_dims(h_w3.size()), 256, 0, stream_compute>>>(d_dl3, d_p2, d_dw3, p3);
            conv2d_backward_bias_kernel<<<get_1d_dims(128), 256, 0, stream_compute>>>(d_dl3, d_db3, p3);
                  
            fill_zeros<<<get_1d_dims(s_l2), 256, 0, stream_compute>>>(d_dl2, s_l2);
            maxpool_backward_kernel<<<get_1d_dims(s_p2), 256, 0, stream_compute>>>(d_dp2, d_l2, d_dl2, B, 16, 16, 128);
            relu_backward_kernel<<<get_1d_dims(s_l2), 256, 0, stream_compute>>>(d_dl2, d_l2, d_dl2, s_l2);

            fill_zeros<<<get_1d_dims(h_w2.size()), 256, 0, stream_compute>>>(d_dw2, h_w2.size());
            fill_zeros<<<get_1d_dims(128), 256, 0, stream_compute>>>(d_db2, 128);
            fill_zeros<<<get_1d_dims(s_p1), 256, 0, stream_compute>>>(d_dp1, s_p1);
            conv2d_backward_input_kernel<<<get_1d_dims(s_p1), 256, 0, stream_compute>>>(d_dl2, d_w2, d_dp1, p2);
            conv2d_backward_weight_kernel<<<get_1d_dims(h_w2.size()), 256, 0, stream_compute>>>(d_dl2, d_p1, d_dw2, p2);
            conv2d_backward_bias_kernel<<<get_1d_dims(128), 256, 0, stream_compute>>>(d_dl2, d_db2, p2);

            fill_zeros<<<get_1d_dims(s_l1), 256, 0, stream_compute>>>(d_dl1, s_l1);
            maxpool_backward_kernel<<<get_1d_dims(s_p1), 256, 0, stream_compute>>>(d_dp1, d_l1, d_dl1, B, 32, 32, 256);
            relu_backward_kernel<<<get_1d_dims(s_l1), 256, 0, stream_compute>>>(d_dl1, d_l1, d_dl1, s_l1);

            fill_zeros<<<get_1d_dims(h_w1.size()), 256, 0, stream_compute>>>(d_dw1, h_w1.size());
            fill_zeros<<<get_1d_dims(256), 256, 0, stream_compute>>>(d_db1, 256);
            conv2d_backward_weight_kernel<<<get_1d_dims(h_w1.size()), 256, 0, stream_compute>>>(d_dl1, d_input[buf_idx], d_dw1, p1);
            conv2d_backward_bias_kernel<<<get_1d_dims(256), 256, 0, stream_compute>>>(d_dl1, d_db1, p1);

            // UPDATE
            update_weights_kernel<<<get_1d_dims(h_w1.size()), 256, 0, stream_compute>>>(d_w1, d_dw1, h_w1.size(), LR);
            update_weights_kernel<<<get_1d_dims(256), 256, 0, stream_compute>>>(d_b1, d_db1, 256, LR);
            update_weights_kernel<<<get_1d_dims(h_w2.size()), 256, 0, stream_compute>>>(d_w2, d_dw2, h_w2.size(), LR);
            update_weights_kernel<<<get_1d_dims(128), 256, 0, stream_compute>>>(d_b2, d_db2, 128, LR);
            update_weights_kernel<<<get_1d_dims(h_w3.size()), 256, 0, stream_compute>>>(d_w3, d_dw3, h_w3.size(), LR);
            update_weights_kernel<<<get_1d_dims(128), 256, 0, stream_compute>>>(d_b3, d_db3, 128, LR);
            update_weights_kernel<<<get_1d_dims(h_w4.size()), 256, 0, stream_compute>>>(d_w4, d_dw4, h_w4.size(), LR);
            update_weights_kernel<<<get_1d_dims(256), 256, 0, stream_compute>>>(d_b4, d_db4, 256, LR);
            update_weights_kernel<<<get_1d_dims(h_w5.size()), 256, 0, stream_compute>>>(d_w5, d_dw5, h_w5.size(), LR);
            update_weights_kernel<<<get_1d_dims(3), 256, 0, stream_compute>>>(d_b5, d_db5, 3, LR);
        }
    }
     
    cudaStreamSynchronize(stream_compute);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "train_gpu_optimize_cuda_streams: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
        cudaFree(d_dl1); cudaFree(d_dp1); cudaFree(d_dl2); cudaFree(d_dp2);
        cudaFree(d_dl3); cudaFree(d_du3); cudaFree(d_dl4); cudaFree(d_du4); cudaFree(d_dout);
    
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_dw1); cudaFree(d_db1);
    cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_dw2); cudaFree(d_db2);
    cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_dw3); cudaFree(d_db3);
    cudaFree(d_w4); cudaFree(d_b4); cudaFree(d_dw4); cudaFree(d_db4);
    cudaFree(d_w5); cudaFree(d_b5); cudaFree(d_dw5); cudaFree(d_db5);
    cudaFree(d_input[0]); cudaFree(d_input[1]);
    cudaFree(d_l1); cudaFree(d_p1); cudaFree(d_l2); cudaFree(d_p2);
    cudaFree(d_l3); cudaFree(d_u3); cudaFree(d_l4); cudaFree(d_u4); cudaFree(d_out);
    cudaFree(d_loss);
    
    return 0;
}
