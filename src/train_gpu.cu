#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>      // For sqrt
#include "cifar10_dataset.h"
#include "kernels.h"  // The header we just created
#include <cuda_runtime.h>

// --- Error Checking Helper ---
void checkCudaErrors(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " (Code: " << code << ")\n";
        exit(code);
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

// --- HOST LOSS FUNCTION (ASSUMED TO USE KERNEL INTERNALLY) ---
// This function needs to handle the kernel launch, synchronization, and H->D copy of the result.
float mse_loss_kernel(float* pred, float* target, size_t size); 
void mse_backward_kernel(float* pred, float* target, float* grad_out, size_t size);


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

    size_t size_input   = (size_t)BATCH * 32 * 32 * 3;
    size_t size_l1_out  = (size_t)BATCH * 32 * 32 * 256;
    size_t size_l1_pool = (size_t)BATCH * 16 * 16 * 256;
    size_t size_l2_out  = (size_t)BATCH * 16 * 16 * 128;
    size_t size_latent  = (size_t)BATCH * 8 * 8 * 128;

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
    allocate_device_buffer(d_l3_up, size_l2_out);
    allocate_device_buffer(d_l4_out, size_l1_pool);
    allocate_device_buffer(d_l4_up, size_l1_out);
    allocate_device_buffer(d_final_out, size_input);

    allocate_device_buffer(d_d_input, size_input);
    allocate_device_buffer(d_d_l1_out, size_l1_out);
    allocate_device_buffer(d_d_l1_pool, size_l1_pool);
    allocate_device_buffer(d_d_l2_out, size_l2_out);
    allocate_device_buffer(d_d_latent, size_latent);
    allocate_device_buffer(d_d_l3_out, size_latent);
    allocate_device_buffer(d_d_l3_up, size_l2_out);
    allocate_device_buffer(d_d_l4_out, size_l1_pool);
    allocate_device_buffer(d_d_l4_up, size_l1_out);
    allocate_device_buffer(d_d_final_out, size_input);


    // 3. TRAINING LOOP
    std::cout << "--- START FULL TRAINING (CUDA) ---\n";
    
    ConvParam_G p1 = {BATCH, 32, 32, 3,   32, 32, 256, 3, 1, 1};
    ConvParam_G p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1};
    ConvParam_G p3 = {BATCH, 8, 8, 128,   8, 8, 128,   3, 1, 1};
    ConvParam_G p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1};
    ConvParam_G p5 = {BATCH, 32, 32, 256, 32, 32, 3,   3, 1, 1};

    int num_batches = MAX_IMAGES / BATCH;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // A. Copy Batch to Device
            size_t offset = (size_t)b * size_input;
            checkCudaErrors(cudaMemcpy(d_input, 
                                     dataset.get_train_images_ptr() + offset, 
                                     size_input * sizeof(float), 
                                     cudaMemcpyHostToDevice));

            // B. FORWARD PASS (Direct Kernel Launches)
            conv2d_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_input, d_w1, d_b1, d_l1_out, p1);
            checkCudaErrors(cudaGetLastError());
            relu_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_l1_out, size_l1_out);
            checkCudaErrors(cudaGetLastError());
            maxpool_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_l1_out, d_l1_pool, BATCH, 32, 32, 256);
            checkCudaErrors(cudaGetLastError());

            conv2d_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_l1_pool, d_w2, d_b2, d_l2_out, p2);
            checkCudaErrors(cudaGetLastError());
            relu_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_l2_out, size_l2_out);
            checkCudaErrors(cudaGetLastError());
            maxpool_kernel<<<get_1d_dims(size_latent), 256>>>(d_l2_out, d_latent, BATCH, 16, 16, 128);
            checkCudaErrors(cudaGetLastError());
            
            conv2d_kernel<<<get_1d_dims(size_latent), 256>>>(d_latent, d_w3, d_b3, d_l3_out, p3);
            checkCudaErrors(cudaGetLastError());
            relu_kernel<<<get_1d_dims(size_latent), 256>>>(d_l3_out, size_latent);
            checkCudaErrors(cudaGetLastError());
            upsample_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_l3_out, d_l3_up, BATCH, 8, 8, 128);
            checkCudaErrors(cudaGetLastError());

            conv2d_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_l3_up, d_w4, d_b4, d_l4_out, p4);
            checkCudaErrors(cudaGetLastError());
            relu_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_l4_out, size_l1_pool);
            checkCudaErrors(cudaGetLastError());
            upsample_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_l4_out, d_l4_up, BATCH, 16, 16, 256);
            checkCudaErrors(cudaGetLastError());

            conv2d_kernel<<<get_1d_dims(size_input), 256>>>(d_l4_up, d_w5, d_b5, d_final_out, p5);
            checkCudaErrors(cudaGetLastError());

            // C. Loss (Assumed Host Wrapper with internal sync/copy)
            float loss = mse_loss_kernel(d_final_out, d_input, size_input);
            total_loss += loss;
            
            // D. BACKWARD PASS (Direct Kernel Launches)
            mse_backward_kernel<<<get_1d_dims(size_input), 256>>>(d_final_out, d_input, d_d_final_out, size_input);
            checkCudaErrors(cudaGetLastError());

            conv2d_backward_kernel<<<get_1d_dims(size_input), 256>>>(d_d_final_out, d_l4_up, d_w5, d_d_l4_up, d_dw5, d_db5, p5);
            checkCudaErrors(cudaGetLastError());
            
            upsample_backward_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_d_l4_up, d_d_l4_out, BATCH, 16, 16, 256);
            checkCudaErrors(cudaGetLastError());
            relu_backward_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_d_l4_out, d_l4_out, d_d_l4_out, size_l1_pool);
            checkCudaErrors(cudaGetLastError());
            conv2d_backward_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_d_l4_out, d_l3_up, d_w4, d_d_l3_up, d_dw4, d_db4, p4);
            checkCudaErrors(cudaGetLastError());

            upsample_backward_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_d_l3_up, d_d_l3_out, BATCH, 8, 8, 128);
            checkCudaErrors(cudaGetLastError());
            relu_backward_kernel<<<get_1d_dims(size_latent), 256>>>(d_d_l3_out, d_l3_out, d_d_l3_out, size_latent);
            checkCudaErrors(cudaGetLastError());
            conv2d_backward_kernel<<<get_1d_dims(size_latent), 256>>>(d_d_l3_out, d_latent, d_w3, d_d_latent, d_dw3, d_db3, p3);
            checkCudaErrors(cudaGetLastError());

            maxpool_backward_kernel<<<get_1d_dims(size_latent), 256>>>(d_d_latent, d_l2_out, d_d_l2_out, BATCH, 16, 16, 128);
            checkCudaErrors(cudaGetLastError());
            relu_backward_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_d_l2_out, d_l2_out, d_d_l2_out, size_l2_out);
            checkCudaErrors(cudaGetLastError());
            conv2d_backward_kernel<<<get_1d_dims(size_l2_out), 256>>>(d_d_l2_out, d_l1_pool, d_w2, d_d_l1_pool, d_dw2, d_db2, p2);
            checkCudaErrors(cudaGetLastError());

            maxpool_backward_kernel<<<get_1d_dims(size_l1_pool), 256>>>(d_d_l1_pool, d_l1_out, d_d_l1_out, BATCH, 32, 32, 256);
            checkCudaErrors(cudaGetLastError());
            relu_backward_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_d_l1_out, d_l1_out, d_d_l1_out, size_l1_out);
            checkCudaErrors(cudaGetLastError());
            conv2d_backward_kernel<<<get_1d_dims(size_l1_out), 256>>>(d_d_l1_out, d_input, d_w1, d_d_input, d_dw1, d_db1, p1);
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
        std::chrono::duration<double> elapsed_epoch = end_epoch - start_total; // Using start_total for continuous time or define start_epoch at the loop beginning
        
            std::cout << "\nEpoch " << epoch + 1 << " Done. Avg Loss: " << total_loss / num_batches 
                    << " | Time: " << elapsed_epoch.count() << "s\n";
        } // End of epoch loop

        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_total = end_total - start_total;
        std::cout << "\n--- Training Complete ---\n";
        std::cout << "Total Training Time: " << elapsed_total.count() << " seconds\n";
    

    // 4. COPY FINAL WEIGHTS BACK TO HOST & SAVE (Omitted for brevity)
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