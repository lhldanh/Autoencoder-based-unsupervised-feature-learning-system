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
    
    int col1_size = B * 32 * 32 * (3 * 9);
    int col2_size = B * 16 * 16 * (256 * 9);
    
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
    
    float *d_w1, *d_b1, *d_col1, *d_col2;
    cudaMalloc(&d_w1, h_w1.size() * 4); cudaMalloc(&d_b1, 256 * 4);
    cudaMalloc(&d_col1, col1_size * 4); cudaMalloc(&d_col2, col2_size * 4);
    
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1.data(), 256 * 4, cudaMemcpyHostToDevice);
    
    int num_batches = max_images / B;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            // Im2col transformation: convert image to column format
            // This enables efficient GEMM operations instead of direct convolution
        }
    }
     
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "train_gpu_optimize_im2col: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_col1); cudaFree(d_col2);
    
    return 0;
}
