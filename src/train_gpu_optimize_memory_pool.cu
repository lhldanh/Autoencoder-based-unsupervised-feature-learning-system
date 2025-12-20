#include <iostream>
#include <vector>
#include <chrono>
#include "cifar10_dataset.h"
#include "kernel.h"

class MemoryPool {
private:
    std::vector<float*> buffers;
    size_t total_size = 0;
public:
    float* alloc(size_t size_bytes) {
        float* ptr;
        cudaMalloc(&ptr, size_bytes);
        buffers.push_back(ptr);
        total_size += size_bytes;
        return ptr;
    }
    size_t get_total() { return total_size; }
    ~MemoryPool() {
        for (auto ptr : buffers) cudaFree(ptr);
    }
};

int main() {
    const int B = 64, EPOCHS = 5;
    const int max_images = 96;
    
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin");
    dataset.load_data();
    if (dataset.get_num_train() == 0) return 1;
    
    MemoryPool pool;
    
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
    
    float *d_w1, *d_b1;
    float *d_w2, *d_b2;
    float *d_w3, *d_b3;
    float *d_w4, *d_b4;
    float *d_w5, *d_b5;
    
    d_w1 = pool.alloc(h_w1.size() * 4); d_b1 = pool.alloc(256 * 4);
    d_w2 = pool.alloc(h_w2.size() * 4); d_b2 = pool.alloc(128 * 4);
    d_w3 = pool.alloc(h_w3.size() * 4); d_b3 = pool.alloc(128 * 4);
    d_w4 = pool.alloc(h_w4.size() * 4); d_b4 = pool.alloc(256 * 4);
    d_w5 = pool.alloc(h_w5.size() * 4); d_b5 = pool.alloc(3 * 4);
    
    float *d_input;
    d_input = pool.alloc(s_in * 4);
    
    int *d_idx1 = (int*)pool.alloc(s_p1 * 4);
    int *d_idx2 = (int*)pool.alloc(s_p2 * 4);
    
    float* d_loss = pool.alloc(4);
    
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
    
    int num_batches = max_images / B;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            float* curr_input = d_input;
            
            // Copy input
            cudaMemcpy(curr_input, dataset.get_train_images_ptr() + batch * s_in, s_in * 4, cudaMemcpyHostToDevice);
            
            // Basic forward pass (simplified - using kernel.h functions)
            // Note: This would need appropriate im2col, conv, pooling, upsampling kernels
        }
    }
     
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "train_gpu_optimize_memory_pool: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
    
    return 0;
}
