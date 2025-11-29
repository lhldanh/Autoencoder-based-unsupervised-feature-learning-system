#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono> // Thư viện đo thời gian
#include "cifar10_dataset.h"
#include "kernels.h" // Header chứa tên hàm chung (không có _cpu)

// Hàm khởi tạo Xavier
void init_random(std::vector<float>& vec, int fan_in, int fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> d(-limit, limit);
    for (auto& x : vec) x = d(gen);
}

// Hàm lưu file
void save_weights(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        uint32_t size = data.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        file.close();
        // std::cout << "Saved: " << filename << "\n";
    } else {
        std::cerr << "Error saving: " << filename << "\n";
    }
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

    // 2. DEFINE ARCHITECTURE & MEMORY
    std::cout << "Allocating memory...\n";
    // Layer 1: Conv 3->256
    std::vector<float> w1(256*3*3*3);      std::vector<float> dw1(w1.size());
    std::vector<float> b1(256, 0.0f);      std::vector<float> db1(256);
    init_random(w1, 3*3*3, 256*3*3);

    // Layer 2: Conv 256->128
    std::vector<float> w2(128*256*3*3);    std::vector<float> dw2(w2.size());
    std::vector<float> b2(128, 0.0f);      std::vector<float> db2(128);
    init_random(w2, 256*3*3, 128*3*3);

    // Layer 3 (Decoder): Conv 128->128
    std::vector<float> w3(128*128*3*3);    std::vector<float> dw3(w3.size());
    std::vector<float> b3(128, 0.0f);      std::vector<float> db3(128);
    init_random(w3, 128*3*3, 128*3*3);

    // Layer 4 (Decoder): Conv 128->256
    std::vector<float> w4(256*128*3*3);    std::vector<float> dw4(w4.size());
    std::vector<float> b4(256, 0.0f);      std::vector<float> db4(256);
    init_random(w4, 128*3*3, 256*3*3);

    // Layer 5 (Output): Conv 256->3
    std::vector<float> w5(3*256*3*3);      std::vector<float> dw5(w5.size());
    std::vector<float> b5(3, 0.0f);        std::vector<float> db5(3);
    init_random(w5, 256*3*3, 3*3*3);

    // ACTIVATION & GRADIENT BUFFERS
    std::vector<float> input_batch(BATCH * 32 * 32 * 3);
    
    // L1
    std::vector<float> l1_out(BATCH*32*32*256);   std::vector<float> d_l1_out(l1_out.size());
    std::vector<float> l1_pool(BATCH*16*16*256);  std::vector<float> d_l1_pool(l1_pool.size());

    // L2
    std::vector<float> l2_out(BATCH*16*16*128);   std::vector<float> d_l2_out(l2_out.size());
    std::vector<float> latent(BATCH*8*8*128);     std::vector<float> d_latent(latent.size());

    // L3
    std::vector<float> l3_out(BATCH*8*8*128);     std::vector<float> d_l3_out(l3_out.size());
    std::vector<float> l3_up(BATCH*16*16*128);    std::vector<float> d_l3_up(l3_up.size());

    // L4
    std::vector<float> l4_out(BATCH*16*16*256);   std::vector<float> d_l4_out(l4_out.size());
    std::vector<float> l4_up(BATCH*32*32*256);    std::vector<float> d_l4_up(l4_up.size());

    // L5
    std::vector<float> final_out(BATCH*32*32*3);  std::vector<float> d_final_out(final_out.size());
    std::vector<float> d_input(input_batch.size()); 

    // 3. TRAINING LOOP
    std::cout << "--- START FULL TRAINING (CPU) ---\n";
    std::cout << "Architecture: 5 Layers (2 Encoder, 3 Decoder)\n";
    std::cout << "Config: " << EPOCHS << " Epochs, ~" << MAX_IMAGES << " Images\n";
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    int num_batches = MAX_IMAGES / BATCH;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // Load Batch
            int image_size = 32 * 32 * 3;
            size_t offset = (size_t)b * BATCH * image_size;
            std::copy(dataset.get_train_images_ptr() + offset, 
                      dataset.get_train_images_ptr() + offset + input_batch.size(), 
                      input_batch.begin());
            
            // --- FORWARD PASS (Generic Names) ---
            ConvParam p1 = {BATCH, 32, 32, 3, 32, 32, 256, 3, 1, 1};
            conv2d(input_batch.data(), w1.data(), b1.data(), l1_out.data(), p1);
            relu(l1_out.data(), l1_out.size());
            maxpool(l1_out.data(), l1_pool.data(), BATCH, 32, 32, 256);

            ConvParam p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1};
            conv2d(l1_pool.data(), w2.data(), b2.data(), l2_out.data(), p2);
            relu(l2_out.data(), l2_out.size());
            maxpool(l2_out.data(), latent.data(), BATCH, 16, 16, 128);

            ConvParam p3 = {BATCH, 8, 8, 128, 8, 8, 128, 3, 1, 1};
            conv2d(latent.data(), w3.data(), b3.data(), l3_out.data(), p3);
            relu(l3_out.data(), l3_out.size());
            upsample(l3_out.data(), l3_up.data(), BATCH, 8, 8, 128);

            ConvParam p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1};
            conv2d(l3_up.data(), w4.data(), b4.data(), l4_out.data(), p4);
            relu(l4_out.data(), l4_out.size());
            upsample(l4_out.data(), l4_up.data(), BATCH, 16, 16, 256);

            ConvParam p5 = {BATCH, 32, 32, 256, 32, 32, 3, 3, 1, 1};
            conv2d(l4_up.data(), w5.data(), b5.data(), final_out.data(), p5);

            // Loss
            float loss = mse_loss(final_out.data(), input_batch.data(), final_out.size());
            total_loss += loss;
            std::cout << "Batch " << b+1 << "/" << num_batches << " | Loss: " << loss << " (Backprop...)\r" << std::flush;

            // --- BACKWARD PASS (Generic Names) ---
            for(size_t i=0; i<final_out.size(); ++i) 
                d_final_out[i] = 2.0f * (final_out[i] - input_batch[i]) / final_out.size();

            conv2d_backward(d_final_out.data(), l4_up.data(), w5.data(), 
                            d_l4_up.data(), dw5.data(), db5.data(), p5);

            upsample_backward(d_l4_up.data(), d_l4_out.data(), BATCH, 16, 16, 256);
            relu_backward(d_l4_out.data(), l4_out.data(), d_l4_out.data(), l4_out.size());
            conv2d_backward(d_l4_out.data(), l3_up.data(), w4.data(), 
                            d_l3_up.data(), dw4.data(), db4.data(), p4);

            upsample_backward(d_l3_up.data(), d_l3_out.data(), BATCH, 8, 8, 128);
            relu_backward(d_l3_out.data(), l3_out.data(), d_l3_out.data(), l3_out.size());
            conv2d_backward(d_l3_out.data(), latent.data(), w3.data(), 
                            d_latent.data(), dw3.data(), db3.data(), p3);

            maxpool_backward(d_latent.data(), l2_out.data(), d_l2_out.data(), BATCH, 16, 16, 128);
            relu_backward(d_l2_out.data(), l2_out.data(), d_l2_out.data(), l2_out.size());
            conv2d_backward(d_l2_out.data(), l1_pool.data(), w2.data(), 
                            d_l1_pool.data(), dw2.data(), db2.data(), p2);

            maxpool_backward(d_l1_pool.data(), l1_out.data(), d_l1_out.data(), BATCH, 32, 32, 256);
            relu_backward(d_l1_out.data(), l1_out.data(), d_l1_out.data(), l1_out.size());
            conv2d_backward(d_l1_out.data(), input_batch.data(), w1.data(), 
                            d_input.data(), dw1.data(), db1.data(), p1);

            // --- UPDATE WEIGHTS (Generic Names) ---
            update_weights(w1.data(), dw1.data(), w1.size(), LR);
            update_weights(b1.data(), db1.data(), b1.size(), LR);
            update_weights(w2.data(), dw2.data(), w2.size(), LR);
            update_weights(b2.data(), db2.data(), b2.size(), LR);
            update_weights(w3.data(), dw3.data(), w3.size(), LR);
            update_weights(b3.data(), db3.data(), b3.size(), LR);
            update_weights(w4.data(), dw4.data(), w4.size(), LR);
            update_weights(b4.data(), db4.data(), b4.size(), LR);
            update_weights(w5.data(), dw5.data(), w5.size(), LR);
            update_weights(b5.data(), db5.data(), b5.size(), LR);
        }

        auto end_epoch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_epoch = end_epoch - start_epoch;
        std::cout << "Epoch " << epoch << " Done. Avg Loss: " << total_loss / num_batches 
                  << " | Time: " << elapsed_epoch.count() << "s\n";
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "\nTotal Training Time: " << elapsed_total.count() << " seconds\n";

    std::cout << "--- SAVING FULL MODEL ---\n";
    save_weights("../weights/enc_w1.bin", w1); save_weights("../weights/enc_b1.bin", b1);
    save_weights("../weights/enc_w2.bin", w2); save_weights("../weights/enc_b2.bin", b2);
    save_weights("../weights/dec_w3.bin", w3); save_weights("../weights/dec_b3.bin", b3);
    save_weights("../weights/dec_w4.bin", w4); save_weights("../weights/dec_b4.bin", b4);
    save_weights("../weights/dec_w5.bin", w5); save_weights("../weights/dec_b5.bin", b5);

    return 0;
}