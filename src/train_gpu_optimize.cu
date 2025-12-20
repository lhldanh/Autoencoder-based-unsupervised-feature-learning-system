#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "cifar10_dataset.h"
#include "optimize_kernel.h"

int main() {
    const int B = 64, EPOCHS = 20;
    const float LR = 0.001f;
    
    std::cout << "=== CUDA Autoencoder (Merged Bias) ===\n\n";
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";
    
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin");
    dataset.load_data();
    if (dataset.get_num_train() == 0) { std::cerr << "No data!\n"; return 1; }
    std::cout << "Images: " << dataset.get_num_train() << "\n\n";
    
    MemoryPool pool;
    
    // Layer dimensions
    int s_in = B * 32 * 32 * 3;
    int s_l1 = B * 32 * 32 * 256, s_p1 = B * 16 * 16 * 256;
    int s_l2 = B * 16 * 16 * 128, s_p2 = B * 8 * 8 * 128;
    int s_l3 = B * 8 * 8 * 128,   s_u3 = B * 16 * 16 * 128;
    int s_l4 = B * 16 * 16 * 256, s_u4 = B * 32 * 32 * 256;
    
    // Col buffer sizes (NOW WITH BIAS COLUMN +1)
    int col1_size = B * 32 * 32 * (3 * 9 + 1);
    int col2_size = B * 16 * 16 * (256 * 9 + 1);
    int col3_size = B * 8 * 8 * (128 * 9 + 1);
    int col4_size = B * 16 * 16 * (128 * 9 + 1);
    int col5_size = B * 32 * 32 * (256 * 9 + 1);
    
    // Weights (NOW INCLUDE BIAS AS EXTRA COLUMN: [out_ch, in_ch * K * K + 1])
    std::vector<float> h_w1(256 * (3 * 9 + 1));      // 256 x 28
    std::vector<float> h_w2(128 * (256 * 9 + 1));    // 128 x 2305
    std::vector<float> h_w3(128 * (128 * 9 + 1));    // 128 x 1153
    std::vector<float> h_w4(256 * (128 * 9 + 1));    // 256 x 1153
    std::vector<float> h_w5(3 * (256 * 9 + 1));      // 3 x 2305
    
    // Initialize weights (bias column will be initialized separately)
    init_random(h_w1, 27, 256);
    init_random(h_w2, 2304, 128);
    init_random(h_w3, 1152, 128);
    init_random(h_w4, 1152, 256);
    init_random(h_w5, 2304, 3);
    
    // Initialize bias columns (last column of each weight matrix)
    for (int i = 0; i < 256; ++i) h_w1[i * (3 * 9 + 1) + (3 * 9)] = 0.0f;
    for (int i = 0; i < 128; ++i) h_w2[i * (256 * 9 + 1) + (256 * 9)] = 0.0f;
    for (int i = 0; i < 128; ++i) h_w3[i * (128 * 9 + 1) + (128 * 9)] = 0.0f;
    for (int i = 0; i < 256; ++i) h_w4[i * (128 * 9 + 1) + (128 * 9)] = 0.0f;
    for (int i = 0; i < 3; ++i) h_w5[i * (256 * 9 + 1) + (256 * 9)] = 0.0f;
    
    // Device memory - weights and gradients
    float *d_w1, *d_dw1;
    float *d_w2, *d_dw2;
    float *d_w3, *d_dw3;
    float *d_w4, *d_dw4;
    float *d_w5, *d_dw5;
    
    d_w1 = pool.alloc(h_w1.size() * 4); d_dw1 = pool.alloc(h_w1.size() * 4);
    d_w2 = pool.alloc(h_w2.size() * 4); d_dw2 = pool.alloc(h_w2.size() * 4);
    d_w3 = pool.alloc(h_w3.size() * 4); d_dw3 = pool.alloc(h_w3.size() * 4);
    d_w4 = pool.alloc(h_w4.size() * 4); d_dw4 = pool.alloc(h_w4.size() * 4);
    d_w5 = pool.alloc(h_w5.size() * 4); d_dw5 = pool.alloc(h_w5.size() * 4);
    
    // Double buffering for input
    float *d_input[2];
    d_input[0] = pool.alloc(s_in * 4);
    d_input[1] = pool.alloc(s_in * 4);
    
    // Forward buffers
    float *d_l1, *d_p1, *d_l2, *d_p2, *d_l3, *d_u3, *d_l4, *d_u4, *d_out;
    d_l1 = pool.alloc(s_l1 * 4); d_p1 = pool.alloc(s_p1 * 4);
    d_l2 = pool.alloc(s_l2 * 4); d_p2 = pool.alloc(s_p2 * 4);
    d_l3 = pool.alloc(s_l3 * 4); d_u3 = pool.alloc(s_u3 * 4);
    d_l4 = pool.alloc(s_l4 * 4); d_u4 = pool.alloc(s_u4 * 4);
    d_out = pool.alloc(s_in * 4);
    
    // Im2col buffers (with bias column)
    float *d_col1, *d_col2, *d_col3, *d_col4, *d_col5;
    d_col1 = pool.alloc(col1_size * 4);
    d_col2 = pool.alloc(col2_size * 4);
    d_col3 = pool.alloc(col3_size * 4);
    d_col4 = pool.alloc(col4_size * 4);
    d_col5 = pool.alloc(col5_size * 4);
    
    // Backward buffers
    float *d_dl1, *d_dp1, *d_dl2, *d_dp2, *d_dl3, *d_du3, *d_dl4, *d_du4, *d_dout;
    float *d_dcol;
    d_dl1 = pool.alloc(s_l1 * 4); d_dp1 = pool.alloc(s_p1 * 4);
    d_dl2 = pool.alloc(s_l2 * 4); d_dp2 = pool.alloc(s_p2 * 4);
    d_dl3 = pool.alloc(s_l3 * 4); d_du3 = pool.alloc(s_u3 * 4);
    d_dl4 = pool.alloc(s_l4 * 4); d_du4 = pool.alloc(s_u4 * 4);
    d_dout = pool.alloc(s_in * 4);
    d_dcol = pool.alloc(col5_size * 4);
    
    int *d_idx1 = (int*)pool.alloc(s_p1 * 4);
    int *d_idx2 = (int*)pool.alloc(s_p2 * 4);
    
    float* d_loss = pool.alloc(4);
    
    std::cout << "Memory: " << pool.get_total() / (1024 * 1024) << " MB\n\n";
    
    // Pinned host memory
    float* h_pinned_input;
    cudaMallocHost(&h_pinned_input, s_in * 4);
    
    // Copy weights to device
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3.data(), h_w3.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4, h_w4.data(), h_w4.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5, h_w5.data(), h_w5.size() * 4, cudaMemcpyHostToDevice);
    
    int num_batches = dataset.get_num_train() / B;
    std::cout << "Training: " << EPOCHS << " epochs, " << num_batches << " batches\n\n";
    
    // Create streams
    cudaStream_t stream_compute, stream_transfer;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_transfer);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto ep_start = std::chrono::high_resolution_clock::now();
        
        cudaMemsetAsync(d_loss, 0, 4, stream_compute);
        
        // Pre-load first batch
        memcpy(h_pinned_input, dataset.get_train_images_ptr(), s_in * 4);
        cudaMemcpyAsync(d_input[0], h_pinned_input, s_in * 4, cudaMemcpyHostToDevice, stream_transfer);
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int curr_buf = batch % 2;
            int next_buf = (batch + 1) % 2;
            float* curr_input = d_input[curr_buf];
            
            // Async load next batch
            if (batch + 1 < num_batches) {
                cudaStreamSynchronize(stream_transfer);
                memcpy(h_pinned_input, dataset.get_train_images_ptr() + (batch + 1) * s_in, s_in * 4);
                cudaMemcpyAsync(d_input[next_buf], h_pinned_input, s_in * 4, 
                               cudaMemcpyHostToDevice, stream_transfer);
            }
            
            if (batch == 0) cudaStreamSynchronize(stream_transfer);
            
            // ========== FORWARD (With merged bias) ==========
            // Layer 1: Conv + ReLU + MaxPool
            im2col_with_bias(curr_input, d_col1, B, 32, 32, 3, 3, 1, 32, 32, stream_compute);
            gemm_nt_relu(d_col1, d_w1, d_l1, B * 32 * 32, 3 * 9 + 1, 256, true, stream_compute);
            maxpool_forward(d_l1, d_p1, d_idx1, B, 32, 32, 256, stream_compute);
            
            // Layer 2: Conv + ReLU + MaxPool
            im2col_with_bias(d_p1, d_col2, B, 16, 16, 256, 3, 1, 16, 16, stream_compute);
            gemm_nt_relu(d_col2, d_w2, d_l2, B * 16 * 16, 256 * 9 + 1, 128, true, stream_compute);
            maxpool_forward(d_l2, d_p2, d_idx2, B, 16, 16, 128, stream_compute);
            
            // Layer 3: Conv + ReLU + Upsample
            im2col_with_bias(d_p2, d_col3, B, 8, 8, 128, 3, 1, 8, 8, stream_compute);
            gemm_nt_relu(d_col3, d_w3, d_l3, B * 8 * 8, 128 * 9 + 1, 128, true, stream_compute);
            upsample_forward(d_l3, d_u3, B, 8, 8, 128, stream_compute);
            
            // Layer 4: Conv + ReLU + Upsample
            im2col_with_bias(d_u3, d_col4, B, 16, 16, 128, 3, 1, 16, 16, stream_compute);
            gemm_nt_relu(d_col4, d_w4, d_l4, B * 16 * 16, 128 * 9 + 1, 256, true, stream_compute);
            upsample_forward(d_l4, d_u4, B, 16, 16, 256, stream_compute);
            
            // Layer 5: Conv (no ReLU)
            im2col_with_bias(d_u4, d_col5, B, 32, 32, 256, 3, 1, 32, 32, stream_compute);
            gemm_nt_relu(d_col5, d_w5, d_out, B * 32 * 32, 256 * 9 + 1, 3, false, stream_compute);
            
            // ========== FUSED LOSS + BACKWARD ==========
            mse_loss_backward_fused(d_out, curr_input, d_dout, d_loss, s_in, stream_compute);
            
            // ========== BACKWARD WITH FUSED KERNELS ==========
            
            // Layer 5 backward (no ReLU - use standard kernels)
            gemm_nn(d_dout, d_w5, d_dcol, B * 32 * 32, 3, 256 * 9 + 1, stream_compute);
            col2im(d_dcol, d_du4, B, 32, 32, 256, 3, 1, 32, 32, stream_compute);
            gemm_tn(d_dout, d_col5, d_dw5, 3, B * 32 * 32, 256 * 9 + 1, stream_compute);
            
            // Layer 4 backward (FUSED: upsample + relu backward)
            fused_upsample_relu_backward(d_du4, d_l4, d_dl4, B, 16, 16, 256, stream_compute);
            gemm_nn(d_dl4, d_w4, d_dcol, B * 16 * 16, 256, 128 * 9 + 1, stream_compute);
            col2im(d_dcol, d_du3, B, 16, 16, 128, 3, 1, 16, 16, stream_compute);
            gemm_tn(d_dl4, d_col4, d_dw4, 256, B * 16 * 16, 128 * 9 + 1, stream_compute);
            
            // Layer 3 backward (FUSED: upsample + relu backward)
            fused_upsample_relu_backward(d_du3, d_l3, d_dl3, B, 8, 8, 128, stream_compute);
            gemm_nn(d_dl3, d_w3, d_dcol, B * 8 * 8, 128, 128 * 9 + 1, stream_compute);
            col2im(d_dcol, d_dp2, B, 8, 8, 128, 3, 1, 8, 8, stream_compute);
            gemm_tn(d_dl3, d_col3, d_dw3, 128, B * 8 * 8, 128 * 9 + 1, stream_compute);
            
            // Layer 2 backward (FUSED: zero + maxpool + relu backward)
            fused_maxpool_relu_backward(d_dp2, d_idx2, d_l2, d_dl2, s_p2, s_l2, stream_compute);
            gemm_nn(d_dl2, d_w2, d_dcol, B * 16 * 16, 128, 256 * 9 + 1, stream_compute);
            col2im(d_dcol, d_dp1, B, 16, 16, 256, 3, 1, 16, 16, stream_compute);
            gemm_tn(d_dl2, d_col2, d_dw2, 128, B * 16 * 16, 256 * 9 + 1, stream_compute);
            
            // Layer 1 backward (FUSED: zero + maxpool + relu backward)
            fused_maxpool_relu_backward(d_dp1, d_idx1, d_l1, d_dl1, s_p1, s_l1, stream_compute);
            gemm_tn(d_dl1, d_col1, d_dw1, 256, B * 32 * 32, 3 * 9 + 1, stream_compute);
            
            // ========== SGD UPDATE (Vectorized) ==========
            sgd_update_vectorized(d_w1, d_dw1, h_w1.size(), LR, stream_compute);
            sgd_update_vectorized(d_w2, d_dw2, h_w2.size(), LR, stream_compute);
            sgd_update_vectorized(d_w3, d_dw3, h_w3.size(), LR, stream_compute);
            sgd_update_vectorized(d_w4, d_dw4, h_w4.size(), LR, stream_compute);
            sgd_update_vectorized(d_w5, d_dw5, h_w5.size(), LR, stream_compute);
        }
        
        float h_loss;
        cudaMemcpyAsync(&h_loss, d_loss, 4, cudaMemcpyDeviceToHost, stream_compute);
        cudaStreamSynchronize(stream_compute);
        
        auto ep_end = std::chrono::high_resolution_clock::now();
        double ep_time = std::chrono::duration<double>(ep_end - ep_start).count();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(6) << h_loss / (num_batches * s_in)
                  << " | Time: " << std::setprecision(2) << ep_time << "s"
                  << " | " << std::setprecision(0) << (num_batches * B) / ep_time << " img/s\n";
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    
    // Save weights
    cudaMemcpy(h_w1.data(), d_w1, h_w1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w2.data(), d_w2, h_w2.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w3.data(), d_w3, h_w3.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w4.data(), d_w4, h_w4.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w5.data(), d_w5, h_w5.size() * 4, cudaMemcpyDeviceToHost);
    
    system("mkdir -p ../weights");
    save_weights("../weights/enc_w1.bin", h_w1);
    save_weights("../weights/enc_w2.bin", h_w2);
    save_weights("../weights/dec_w3.bin", h_w3);
    save_weights("../weights/dec_w4.bin", h_w4);
    save_weights("../weights/dec_w5.bin", h_w5);
    
    cudaFreeHost(h_pinned_input);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
    std::cout << "Saved weights.\n";
    return 0;
}