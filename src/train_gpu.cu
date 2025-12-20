#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>   // For sqrt
#include "cifar10_dataset.h"
#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h> // For malloc/free

int main() {
  // 1. CONFIG & DATA
  int BATCH = 32;
  int EPOCHS = 5;
  int MAX_IMAGES = 96; // Limit number of images for quick testing
  float LR = 0.001f;

  std::string data_path = "../data/cifar-10-batches-bin";  // Changed from "data/cifar-10-batches-bin"
  CIFAR10Dataset dataset(data_path);
  dataset.load_data();
  if (dataset.get_num_train() == 0) return 1;

// ...existing code...

  // --- HOST WEIGHTS AND BIASES ---
    // Encoder: 32x32x3 -> Conv1 -> 32x32x256 -> MaxPool -> 16x16x256
    // 16x16x256 -> Conv2 -> 16x16x128 -> MaxPool -> 8x8x128 (Latent)
  std::vector<float> h_w1(256*3*3*3);   init_random(h_w1, 3*3*3, 256*3*3);
  std::vector<float> h_b1(256, 0.0f);
  std::vector<float> h_w2(128*256*3*3);  init_random(h_w2, 256*3*3, 128*3*3);
  std::vector<float> h_b2(128, 0.0f);
    // Decoder: 8x8x128 -> Conv3 -> 8x8x128 (Conv on latent to extract features)
    // 8x8x128 -> Upsample -> 16x16x128 -> Conv4 -> 16x16x256
    // 16x16x256 -> Upsample -> 32x32x256 -> Conv5 -> 32x32x3
  std::vector<float> h_w3(128*128*3*3);  init_random(h_w3, 128*3*3, 128*3*3);
  std::vector<float> h_b3(128, 0.0f);
  std::vector<float> h_w4(256*128*3*3);  init_random(h_w4, 128*3*3, 256*3*3);
  std::vector<float> h_b4(256, 0.0f);
  std::vector<float> h_w5(3*256*3*3);   init_random(h_w5, 256*3*3, 3*3*3);
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

  size_t size_input  = (size_t)BATCH * 32 * 32 * 3;
  size_t size_l1_out = (size_t)BATCH * 32 * 32 * 256;
  size_t size_l1_pool = (size_t)BATCH * 16 * 16 * 256;
  size_t size_l2_out = (size_t)BATCH * 16 * 16 * 128;
  size_t size_latent = (size_t)BATCH * 8 * 8 * 128;
    // Decoder output sizes
    // d_l3_out is size_latent
    size_t size_l3_up   = (size_t)BATCH * 16 * 16 * 128;
    size_t size_l4_out  = (size_t)BATCH * 16 * 16 * 256;
    size_t size_l4_up   = (size_t)BATCH * 32 * 32 * 256;


  // 2. ALLOCATE AND COPY MEMORY

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
  std::cout << "--- START TRAINING (CUDA) ---\n";
  std::cout << "Batch Size: " << BATCH << ", Epochs: " << EPOCHS << ", Learning Rate: " << LR << "\n" << "Max Images: " << MAX_IMAGES << "\n";
  // ConvParam_G: B, H_in, W_in, C_in, H_out, W_out, C_out, K, S, P
    // Encoder
  ConvParam_G p1 = {BATCH, 32, 32, 3,  32, 32, 256, 3, 1, 1}; // Output: 32x32x256
  ConvParam_G p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1}; // Output: 16x16x128
    // Decoder
  ConvParam_G p3 = {BATCH, 8, 8, 128,  8, 8, 128,  3, 1, 1}; // Output: 8x8x128 (Latent conv)
  ConvParam_G p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1}; // Output: 16x16x256
  ConvParam_G p5 = {BATCH, 32, 32, 256, 32, 32, 3,  3, 1, 1}; // Output: 32x32x3

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
  // std::cout << "\n--- Copying Final Weights to Host and Saving ---\n";

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
  // save_weights("../weights/enc_w1.bin", h_w1); save_weights("../weights/enc_b1.bin", h_b1);
  // save_weights("../weights/enc_w2.bin", h_w2); save_weights("../weights/enc_b2.bin", h_b2);
  // save_weights("../weights/dec_w3.bin", h_w3); save_weights("../weights/dec_b3.bin", h_b3);
  // save_weights("../weights/dec_w4.bin", h_w4); save_weights("../weights/dec_b4.bin", h_b4);
  // save_weights("../weights/dec_w5.bin", h_w5); save_weights("../weights/dec_b5.bin", h_b5);

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