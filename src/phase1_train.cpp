#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include "cifar10_dataset.h"
#include "kernels.h"

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
        std::cout << "Saved: " << filename << "\n";
    } else {
        std::cerr << "Error saving: " << filename << "\n";
    }
}

int main() {
    // 1. CONFIG & DATA
    // Lưu ý: CPU rất chậm với mạng full. Batch=4 để tránh đợi quá lâu.
    int BATCH = 32;
    int EPOCHS = 1;
    float LR = 0.001f;

    std::string data_path = "../data/cifar-10-batches-bin";
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();
    if (dataset.get_num_train() == 0) return 1;

    // 2. DEFINE ARCHITECTURE & MEMORY [cite: 229]
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
    // Input: 32x32x3
    std::vector<float> input_batch(BATCH * 32 * 32 * 3);
    
    // L1: 32x32x256 -> MaxPool -> 16x16x256
    std::vector<float> l1_out(BATCH*32*32*256);   std::vector<float> d_l1_out(l1_out.size());
    std::vector<float> l1_pool(BATCH*16*16*256);  std::vector<float> d_l1_pool(l1_pool.size());

    // L2: 16x16x128 -> MaxPool -> 8x8x128 (Latent)
    std::vector<float> l2_out(BATCH*16*16*128);   std::vector<float> d_l2_out(l2_out.size());
    std::vector<float> latent(BATCH*8*8*128);     std::vector<float> d_latent(latent.size());

    // L3: 8x8x128 -> Upsample -> 16x16x128
    std::vector<float> l3_out(BATCH*8*8*128);     std::vector<float> d_l3_out(l3_out.size());
    std::vector<float> l3_up(BATCH*16*16*128);    std::vector<float> d_l3_up(l3_up.size());

    // L4: 16x16x256 -> Upsample -> 32x32x256
    std::vector<float> l4_out(BATCH*16*16*256);   std::vector<float> d_l4_out(l4_out.size());
    std::vector<float> l4_up(BATCH*32*32*256);    std::vector<float> d_l4_up(l4_up.size());

    // L5: 32x32x3 (Output)
    std::vector<float> final_out(BATCH*32*32*3);  std::vector<float> d_final_out(final_out.size());
    
    // Buffer tạm cho Input Gradient (để hoàn tất chuỗi backward)
    std::vector<float> d_input(input_batch.size()); 

    // 3. TRAINING LOOP
    std::cout << "--- START FULL TRAINING (CPU) ---\n";
    std::cout << "Architecture: 5 Layers (2 Encoder, 3 Decoder)\n";
    
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = dataset.get_num_train() / BATCH;
        for (int b = 0; b < num_batches; ++b) {
            // Load Batch (Demo: lấy batch đầu tiên)
            int image_size = 32 * 32 * 3;
            size_t offset = (size_t)b * BATCH * image_size;
            
            // Copy từ vị trí offset
            std::copy(dataset.get_train_images_ptr() + offset, 
                      dataset.get_train_images_ptr() + offset + input_batch.size(), 
                      input_batch.begin());
            // --- FORWARD PASS ---
            // L1: Conv -> ReLU -> Pool
            ConvParam p1 = {BATCH, 32, 32, 3, 32, 32, 256, 3, 1, 1};
            conv2d_cpu(input_batch.data(), w1.data(), b1.data(), l1_out.data(), p1);
            relu_cpu(l1_out.data(), l1_out.size());
            maxpool_cpu(l1_out.data(), l1_pool.data(), BATCH, 32, 32, 256);

            // L2: Conv -> ReLU -> Pool
            ConvParam p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1};
            conv2d_cpu(l1_pool.data(), w2.data(), b2.data(), l2_out.data(), p2);
            relu_cpu(l2_out.data(), l2_out.size());
            maxpool_cpu(l2_out.data(), latent.data(), BATCH, 16, 16, 128);

            // L3: Conv -> ReLU -> Upsample
            ConvParam p3 = {BATCH, 8, 8, 128, 8, 8, 128, 3, 1, 1};
            conv2d_cpu(latent.data(), w3.data(), b3.data(), l3_out.data(), p3);
            relu_cpu(l3_out.data(), l3_out.size());
            upsample_cpu(l3_out.data(), l3_up.data(), BATCH, 8, 8, 128);

            // L4: Conv -> ReLU -> Upsample
            ConvParam p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1};
            conv2d_cpu(l3_up.data(), w4.data(), b4.data(), l4_out.data(), p4);
            relu_cpu(l4_out.data(), l4_out.size());
            upsample_cpu(l4_out.data(), l4_up.data(), BATCH, 16, 16, 256);

            // L5: Conv Only (No activation as per PDF)
            ConvParam p5 = {BATCH, 32, 32, 256, 32, 32, 3, 3, 1, 1};
            conv2d_cpu(l4_up.data(), w5.data(), b5.data(), final_out.data(), p5);

            // Loss
            float loss = mse_loss_cpu(final_out.data(), input_batch.data(), final_out.size());
            total_loss += loss;
            std::cout << "Batch " << b << " | Loss: " << loss << " (Calculating Backward...)\n";

            // --- BACKWARD PASS (Reverse Order) ---
            
            // 1. Loss Gradient
            for(size_t i=0; i<final_out.size(); ++i) 
                d_final_out[i] = 2.0f * (final_out[i] - input_batch[i]) / final_out.size();

            // 2. Backprop L5
            conv2d_backward_cpu(d_final_out.data(), l4_up.data(), w5.data(), 
                                d_l4_up.data(), dw5.data(), db5.data(), p5);

            // 3. Backprop L4 (Upsample -> ReLU -> Conv)
            upsample_backward_cpu(d_l4_up.data(), d_l4_out.data(), BATCH, 16, 16, 256);
            relu_backward_cpu(d_l4_out.data(), l4_out.data(), d_l4_out.data(), l4_out.size());
            conv2d_backward_cpu(d_l4_out.data(), l3_up.data(), w4.data(), 
                                d_l3_up.data(), dw4.data(), db4.data(), p4);

            // 4. Backprop L3 (Upsample -> ReLU -> Conv)
            upsample_backward_cpu(d_l3_up.data(), d_l3_out.data(), BATCH, 8, 8, 128);
            relu_backward_cpu(d_l3_out.data(), l3_out.data(), d_l3_out.data(), l3_out.size());
            conv2d_backward_cpu(d_l3_out.data(), latent.data(), w3.data(), 
                                d_latent.data(), dw3.data(), db3.data(), p3);

            // 5. Backprop L2 (MaxPool -> ReLU -> Conv)
            maxpool_backward_cpu(d_latent.data(), l2_out.data(), d_l2_out.data(), BATCH, 16, 16, 128);
            relu_backward_cpu(d_l2_out.data(), l2_out.data(), d_l2_out.data(), l2_out.size());
            conv2d_backward_cpu(d_l2_out.data(), l1_pool.data(), w2.data(), 
                                d_l1_pool.data(), dw2.data(), db2.data(), p2);

            // 6. Backprop L1 (MaxPool -> ReLU -> Conv)
            maxpool_backward_cpu(d_l1_pool.data(), l1_out.data(), d_l1_out.data(), BATCH, 32, 32, 256);
            relu_backward_cpu(d_l1_out.data(), l1_out.data(), d_l1_out.data(), l1_out.size());
            conv2d_backward_cpu(d_l1_out.data(), input_batch.data(), w1.data(), 
                                d_input.data(), dw1.data(), db1.data(), p1);

            // --- UPDATE WEIGHTS ---
            update_weights_cpu(w1.data(), dw1.data(), w1.size(), LR);
            update_weights_cpu(b1.data(), db1.data(), b1.size(), LR);
            update_weights_cpu(w2.data(), dw2.data(), w2.size(), LR);
            update_weights_cpu(b2.data(), db2.data(), b2.size(), LR);
            update_weights_cpu(w3.data(), dw3.data(), w3.size(), LR);
            update_weights_cpu(b3.data(), db3.data(), b3.size(), LR);
            update_weights_cpu(w4.data(), dw4.data(), w4.size(), LR);
            update_weights_cpu(b4.data(), db4.data(), b4.size(), LR);
            update_weights_cpu(w5.data(), dw5.data(), w5.size(), LR);
            update_weights_cpu(b5.data(), db5.data(), b5.size(), LR);
        }
        std::cout << "Epoch " << epoch << " Done.\n";
    }

    // 4. SAVE ALL WEIGHTS (10 Files)
    std::cout << "--- SAVING FULL MODEL ---\n";
    save_weights("../weights/enc_w1.bin", w1); save_weights("../weights/enc_b1.bin", b1);
    save_weights("../weights/enc_w2.bin", w2); save_weights("../weights/enc_b2.bin", b2);
    save_weights("../weights/dec_w3.bin", w3); save_weights("../weights/dec_b3.bin", b3);
    save_weights("../weights/dec_w4.bin", w4); save_weights("../weights/dec_b4.bin", b4);
    save_weights("../weights/dec_w5.bin", w5); save_weights("../weights/dec_b5.bin", b5);

    return 0;
}