#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include "cifar10_dataset.h"
#include "kernels.h" // Includes ConvParam struct and all function prototypes
#include <cuda_runtime.h> // For cudaMalloc/cudaMemcpy

// --- Custom CUDA Error Checking Utility ---
// Definition for checkCudaErrors, which was undefined in the original compilation output.
void checkCudaErrors(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " (Code: " << code << ")\n";
        exit(code);
    }
}

// --- Forward Declarations for Missing Kernel Prototypes ---
// These are added to resolve the "identifier is undefined" errors for the loss functions,
// as they are expected to be implemented in a separate CUDA file (linked later).
extern "C" float mse_loss(float* output, float* target, size_t size);
extern "C" void mse_backward(float* output, float* target, float* grad_out, size_t size);


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
    int EPOCHS = 50;
    int MAX_IMAGES = 100; // Giới hạn số ảnh để chạy nhanh trên CPU
    float LR = 0.001f;

    std::string data_path = "../data/cifar-10-batches-bin";
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();
    if (dataset.get_num_train() == 0) return 1;

    // --- HOST WEIGHTS AND BIASES (H_ stands for Host) ---
    // These vectors are kept on the host to hold the initial values and final results.
    std::vector<float> h_w1(256*3*3*3);      init_random(h_w1, 3*3*3, 256*3*3);
    std::vector<float> h_b1(256, 0.0f);
    std::vector<float> h_w2(128*256*3*3);    init_random(h_w2, 256*3*3, 128*3*3);
    std::vector<float> h_b2(128, 0.0f);
    std::vector<float> h_w3(128*128*3*3);    init_random(h_w3, 128*3*3, 128*3*3);
    std::vector<float> h_b3(128, 0.0f);
    std::vector<float> h_w4(256*128*3*3);    init_random(h_w4, 128*3*3, 256*3*3);
    std::vector<float> h_b4(256, 0.0f);
    std::vector<float> h_w5(3*256*3*3);      init_random(h_w5, 256*3*3, 3*3*3);
    std::vector<float> h_b5(3, 0.0f);

    // --- DEVICE POINTERS (D_ stands for Device) ---
    float *d_w1, *d_b1, *d_dw1, *d_db1;
    float *d_w2, *d_b2, *d_dw2, *d_db2;
    float *d_w3, *d_b3, *d_dw3, *d_db3;
    float *d_w4, *d_b4, *d_dw4, *d_db4;
    float *d_w5, *d_b5, *d_dw5, *d_db5;

    // --- DEVICE ACTIVATION & GRADIENT BUFFERS ---
    size_t input_size = (size_t)BATCH * 32 * 32 * 3;
    size_t l1_out_size = (size_t)BATCH * 32 * 32 * 256;
    size_t l1_pool_size = (size_t)BATCH * 16 * 16 * 256;
    size_t l2_out_size = (size_t)BATCH * 16 * 16 * 128;
    size_t latent_size = (size_t)BATCH * 8 * 8 * 128;
    size_t final_out_size = input_size;

    float *d_input_batch, *d_l1_out, *d_l1_pool, *d_l2_out, *d_latent;
    float *d_l3_out, *d_l3_up, *d_l4_out, *d_l4_up, *d_final_out;
    float *d_d_input, *d_d_l1_out, *d_d_l1_pool, *d_d_l2_out, *d_d_latent;
    float *d_d_l3_out, *d_d_l3_up, *d_d_l4_out, *d_d_l4_up, *d_d_final_out;
    float *d_input_h_buffer; // Temporary buffer to copy host data for each batch

    // 2. ALLOCATE DEVICE MEMORY
    std::cout << "Allocating and copying initial weights to GPU...\n";
    
    // Weights and Biases (Allocate and Copy H->D)
    allocate_and_copy(d_w1, h_w1); allocate_and_copy(d_b1, h_b1);
    allocate_and_copy(d_w2, h_w2); allocate_and_copy(d_b2, h_b2);
    allocate_and_copy(d_w3, h_w3); allocate_and_copy(d_b3, h_b3);
    allocate_and_copy(d_w4, h_w4); allocate_and_copy(d_b4, h_b4);
    allocate_and_copy(d_w5, h_w5); allocate_and_copy(d_b5, h_b5);

    // Gradient Buffers (Allocate only)
    allocate_device_buffer(d_dw1, h_w1.size()); allocate_device_buffer(d_db1, h_b1.size());
    allocate_device_buffer(d_dw2, h_w2.size()); allocate_device_buffer(d_db2, h_b2.size());
    allocate_device_buffer(d_dw3, h_w3.size()); allocate_device_buffer(d_db3, h_b3.size());
    allocate_device_buffer(d_dw4, h_w4.size()); allocate_device_buffer(d_db4, h_b4.size());
    allocate_device_buffer(d_dw5, h_w5.size()); allocate_device_buffer(d_db5, h_b5.size());
    
    // Forward Buffers (Allocate only)
    allocate_device_buffer(d_input_h_buffer, input_size); // Host buffer for batch data
    allocate_device_buffer(d_input_batch, input_size);
    allocate_device_buffer(d_l1_out, l1_out_size);
    allocate_device_buffer(d_l1_pool, l1_pool_size);
    allocate_device_buffer(d_l2_out, l2_out_size);
    allocate_device_buffer(d_latent, latent_size);
    allocate_device_buffer(d_l3_out, latent_size); // l3_out and latent are same size
    allocate_device_buffer(d_l3_up, l2_out_size);
    allocate_device_buffer(d_l4_out, l1_out_size);
    allocate_device_buffer(d_l4_up, input_size);
    allocate_device_buffer(d_final_out, final_out_size);

    // Backward Buffers (Allocate only)
    allocate_device_buffer(d_d_input, input_size);
    allocate_device_buffer(d_d_l1_out, l1_out_size);
    allocate_device_buffer(d_d_l1_pool, l1_pool_size);
    allocate_device_buffer(d_d_l2_out, l2_out_size);
    allocate_device_buffer(d_d_latent, latent_size);
    allocate_device_buffer(d_d_l3_out, latent_size);
    allocate_device_buffer(d_d_l3_up, l2_out_size);
    allocate_device_buffer(d_d_l4_out, l1_out_size);
    allocate_device_buffer(d_d_l4_up, input_size);
    allocate_device_buffer(d_d_final_out, final_out_size);


    // 3. TRAINING LOOP
    std::cout << "--- START FULL TRAINING (CUDA) ---\n";
    std::cout << "Architecture: 5 Layers (2 Encoder, 3 Decoder) [Image of Convolutional Autoencoder Architecture]\n";
    std::cout << "Config: " << EPOCHS << " Epochs, ~" << MAX_IMAGES << " Images\n";
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    int num_batches = MAX_IMAGES / BATCH;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // 3a. Copy Batch to Device
            size_t offset = (size_t)b * BATCH * 32 * 32 * 3;
            // Use d_input_h_buffer as a temporary host vector to read from dataset
            checkCudaErrors(cudaMemcpy(d_input_h_buffer, 
                                     dataset.get_train_images_ptr() + offset, 
                                     input_size * sizeof(float), 
                                     cudaMemcpyHostToDevice));
            
            // Note: The first input buffer (d_input_batch) needs the batch data
            // We use the temporary buffer as input to the first layer.
            d_input_batch = d_input_h_buffer; 

            // --- FORWARD PASS (USING DEVICE POINTERS) ---
            ConvParam p1 = {BATCH, 32, 32, 3, 32, 32, 256, 3, 1, 1};
            conv2d(d_input_batch, d_w1, d_b1, d_l1_out, p1);
            relu(d_l1_out, l1_out_size);
            maxpool(d_l1_out, d_l1_pool, BATCH, 32, 32, 256);

            ConvParam p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1};
            conv2d(d_l1_pool, d_w2, d_b2, d_l2_out, p2);
            relu(d_l2_out, l2_out_size);
            maxpool(d_l2_out, d_latent, BATCH, 16, 16, 128);

            ConvParam p3 = {BATCH, 8, 8, 128, 8, 8, 128, 3, 1, 1};
            conv2d(d_latent, d_w3, d_b3, d_l3_out, p3);
            relu(d_l3_out, latent_size);
            upsample(d_l3_out, d_l3_up, BATCH, 8, 8, 128);

            ConvParam p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1};
            conv2d(d_l3_up, d_w4, d_b4, d_l4_out, p4);
            relu(d_l4_out, l1_out_size);
            upsample(d_l4_out, d_l4_up, BATCH, 16, 16, 256);

            ConvParam p5 = {BATCH, 32, 32, 256, 32, 32, 3, 3, 1, 1};
            conv2d(d_l4_up, d_w5, d_b5, d_final_out, p5);

            // Loss: Calculated on Device, result copied back to Host
            float loss = mse_loss(d_final_out, d_input_batch, final_out_size);
            total_loss += loss;
            std::cout << "Batch " << b+1 << "/" << num_batches << " | Loss: " << loss << " (Backprop...)\r" << std::flush;

            // --- BACKWARD PASS (USING DEVICE POINTERS) ---
            
            // 1. MSE Gradient (d_final_out)
            mse_backward(d_final_out, d_input_batch, d_d_final_out, final_out_size);

            // 2. Decoder Layers (L5 -> L3)
            conv2d_backward(d_d_final_out, d_l4_up, d_w5, d_d_l4_up, d_dw5, d_db5, p5);

            upsample_backward(d_d_l4_up, d_d_l4_out, BATCH, 16, 16, 256);
            relu_backward(d_d_l4_out, d_l4_out, d_d_l4_out, l1_out_size);
            conv2d_backward(d_d_l4_out, d_l3_up, d_w4, d_d_l3_up, d_dw4, d_db4, p4);

            upsample_backward(d_d_l3_up, d_d_l3_out, BATCH, 8, 8, 128);
            relu_backward(d_d_l3_out, d_l3_out, d_d_l3_out, latent_size);
            conv2d_backward(d_d_l3_out, d_latent, d_w3, d_d_latent, d_dw3, d_db3, p3);

            // 3. Encoder Layers (L2 -> L1)
            maxpool_backward(d_d_latent, d_l2_out, d_d_l2_out, BATCH, 16, 16, 128);
            relu_backward(d_d_l2_out, d_l2_out, d_d_l2_out, l2_out_size);
            conv2d_backward(d_d_l2_out, d_l1_pool, d_w2, d_d_l1_pool, d_dw2, d_db2, p2);

            maxpool_backward(d_d_l1_pool, d_l1_out, d_d_l1_out, BATCH, 32, 32, 256);
            relu_backward(d_d_l1_out, d_l1_out, d_d_l1_out, l1_out_size);
            conv2d_backward(d_d_l1_out, d_input_batch, d_w1, d_d_input, d_dw1, d_db1, p1);

            // --- UPDATE WEIGHTS (USING DEVICE POINTERS) ---
            update_weights(d_w1, d_dw1, h_w1.size(), LR);
            update_weights(d_b1, d_db1, h_b1.size(), LR);
            update_weights(d_w2, d_dw2, h_w2.size(), LR);
            update_weights(d_b2, d_db2, h_b2.size(), LR);
            update_weights(d_w3, d_dw3, h_w3.size(), LR);
            update_weights(d_b3, d_db3, h_b3.size(), LR);
            update_weights(d_w4, d_dw4, h_w4.size(), LR);
            update_weights(d_b4, d_db4, h_b4.size(), LR);
            update_weights(d_w5, d_dw5, h_w5.size(), LR);
            update_weights(d_b5, d_db5, h_b5.size(), LR);
        }

        auto end_epoch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_epoch = end_epoch - start_epoch;
        std::cout << "Epoch " << epoch << " Done. Avg Loss: " << total_loss / num_batches 
                  << " | Time: " << elapsed_epoch.count() << "s\n";
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "\nTotal Training Time: " << elapsed_total.count() << " seconds\n";

    // 4. COPY FINAL WEIGHTS BACK TO HOST
    std::cout << "--- COPYING FINAL WEIGHTS TO HOST AND SAVING ---\n";
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

    save_weights("../weights/enc_w1.bin", h_w1); save_weights("../weights/enc_b1.bin", h_b1);
    save_weights("../weights/enc_w2.bin", h_w2); save_weights("../weights/enc_b2.bin", h_b2);
    save_weights("../weights/dec_w3.bin", h_w3); save_weights("../weights/dec_b3.bin", h_b3);
    save_weights("../weights/dec_w4.bin", h_w4); save_weights("../weights/dec_b4.bin", h_b4);
    save_weights("../weights/dec_w5.bin", h_w5); save_weights("../weights/dec_b5.bin", h_b5);

    // 5. CLEANUP DEVICE MEMORY
    std::cout << "Cleaning up device memory...\n";
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_dw1); cudaFree(d_db1);
    cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_dw2); cudaFree(d_db2);
    cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_dw3); cudaFree(d_db3);
    cudaFree(d_w4); cudaFree(d_b4); cudaFree(d_dw4); cudaFree(d_db4);
    cudaFree(d_w5); cudaFree(d_b5); cudaFree(d_dw5); cudaFree(d_db5);
    
    cudaFree(d_input_h_buffer);
    cudaFree(d_l1_out); cudaFree(d_l1_pool); cudaFree(d_l2_out); cudaFree(d_latent);
    cudaFree(d_l3_out); cudaFree(d_l3_up); cudaFree(d_l4_out); cudaFree(d_l4_up); cudaFree(d_final_out);
    cudaFree(d_d_input); cudaFree(d_d_l1_out); cudaFree(d_d_l1_pool); cudaFree(d_d_l2_out); cudaFree(d_d_latent);
    cudaFree(d_d_l3_out); cudaFree(d_d_l3_up); cudaFree(d_d_l4_out); cudaFree(d_d_l4_up); cudaFree(d_d_final_out);

    return 0;
}