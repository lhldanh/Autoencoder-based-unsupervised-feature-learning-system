#include <iostream>
#include <vector>
#include <chrono>
#include "cifar10_dataset.h"
#include "optimize_kernel.h"

int main() {
    const int B = 64, EPOCHS = 5;
    const int max_images = 96;
    
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin");
    dataset.load_data();
    if (dataset.get_num_train() == 0) return 1;
    
    int s_in = B * 32 * 32 * 3;
    
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
    
    float *d_w1, *d_b1, *d_col1, *d_out;
    cudaMalloc(&d_w1, h_w1.size() * 4); cudaMalloc(&d_b1, 256 * 4);
    cudaMalloc(&d_col1, B * 32 * 32 * 27 * 4);
    cudaMalloc(&d_out, s_in * 4);
    
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1.data(), 256 * 4, cudaMemcpyHostToDevice);
    
    int num_batches = max_images / B;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            // GEMM (General Matrix Multiply) operations
            // Replaces convolution with highly optimized matrix multiplication
            // gemm_nt_bias_relu, gemm_nn, gemm_tn kernels
        }
    }
     
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "train_gpu_optimize_gemm: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_col1); cudaFree(d_out);
    
    return 0;
}
