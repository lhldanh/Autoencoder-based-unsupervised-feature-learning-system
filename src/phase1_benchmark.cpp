#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "cifar10_dataset.h"
#include "kernels.h"

// Hàm khởi tạo trọng số ngẫu nhiên (Xavier initialization đơn giản)
void init_random(std::vector<float>& vec, int fan_in, int fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> d(-limit, limit);
    for (auto& x : vec) x = d(gen);
}

int main() {
    // 1. LOAD DATA
    std::cout << "--- PHASE 1: FULL CPU AUTOENCODER ---" << std::endl;
    std::string data_path = "../data/cifar-10-batches-bin"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();
    
    if (dataset.get_num_train() == 0) return 1;

    int BATCH = 16; // Giữ batch nhỏ để test trên CPU
    std::cout << "Processing Batch Size: " << BATCH << "\n";

    // Prepare Input & Output Buffers
    std::vector<float> input_batch(BATCH * 32 * 32 * 3);
    std::copy(dataset.get_train_images_ptr(), 
              dataset.get_train_images_ptr() + input_batch.size(), 
              input_batch.begin());
    
    // --- KHỞI TẠO MẠNG (ARCHITECTURE) ---
    // Xem chi tiết kiến trúc tại [cite: 174-207]

    // === ENCODER ===
    // Layer 1: Conv(3->256) -> ReLU -> MaxPool (32x32 -> 16x16)
    std::vector<float> enc_w1(256 * 3 * 3 * 3); init_random(enc_w1, 3*3*3, 256*3*3);
    std::vector<float> enc_b1(256, 0.0f);
    std::vector<float> enc_out1(BATCH * 32 * 32 * 256);
    std::vector<float> enc_pool1(BATCH * 16 * 16 * 256);

    // Layer 2: Conv(256->128) -> ReLU -> MaxPool (16x16 -> 8x8) -> LATENT
    std::vector<float> enc_w2(128 * 256 * 3 * 3); init_random(enc_w2, 256*3*3, 128*3*3);
    std::vector<float> enc_b2(128, 0.0f);
    std::vector<float> enc_out2(BATCH * 16 * 16 * 128);
    std::vector<float> latent(BATCH * 8 * 8 * 128);

    // === DECODER ===
    // Layer 3: Conv(128->128) -> ReLU -> UpSample (8x8 -> 16x16)
    std::vector<float> dec_w1(128 * 128 * 3 * 3); init_random(dec_w1, 128*3*3, 128*3*3);
    std::vector<float> dec_b1(128, 0.0f);
    std::vector<float> dec_out1(BATCH * 8 * 8 * 128); // Sau conv
    std::vector<float> dec_up1(BATCH * 16 * 16 * 128); // Sau upsample

    // Layer 4: Conv(128->256) -> ReLU -> UpSample (16x16 -> 32x32)
    std::vector<float> dec_w2(256 * 128 * 3 * 3); init_random(dec_w2, 128*3*3, 256*3*3);
    std::vector<float> dec_b2(256, 0.0f);
    std::vector<float> dec_out2(BATCH * 16 * 16 * 256);
    std::vector<float> dec_up2(BATCH * 32 * 32 * 256);

    // Layer 5 (Output): Conv(256->3) -> Sigmoid/Linear (32x32)
    std::vector<float> dec_w3(3 * 256 * 3 * 3); init_random(dec_w3, 256*3*3, 3*3*3);
    std::vector<float> dec_b3(3, 0.0f);
    std::vector<float> final_output(BATCH * 32 * 32 * 3);

    // --- BENCHMARK START ---
    std::cout << "Starting Forward Pass...\n";
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Encoder L1
    ConvParam p1 = {BATCH, 32, 32, 3, 32, 32, 256, 3, 1, 1};
    conv2d_cpu(input_batch.data(), enc_w1.data(), enc_b1.data(), enc_out1.data(), p1);
    relu_cpu(enc_out1.data(), enc_out1.size());
    maxpool_cpu(enc_out1.data(), enc_pool1.data(), BATCH, 32, 32, 256);

    // 2. Encoder L2
    ConvParam p2 = {BATCH, 16, 16, 256, 16, 16, 128, 3, 1, 1};
    conv2d_cpu(enc_pool1.data(), enc_w2.data(), enc_b2.data(), enc_out2.data(), p2);
    relu_cpu(enc_out2.data(), enc_out2.size());
    maxpool_cpu(enc_out2.data(), latent.data(), BATCH, 16, 16, 128);

    // 3. Decoder L1 (Lưu ý: Input là Latent 8x8)
    // Conv: 8x8 -> 8x8 (Padding=1 giữ nguyên kích thước)
    ConvParam p3 = {BATCH, 8, 8, 128, 8, 8, 128, 3, 1, 1};
    conv2d_cpu(latent.data(), dec_w1.data(), dec_b1.data(), dec_out1.data(), p3);
    relu_cpu(dec_out1.data(), dec_out1.size());
    // Upsample: 8x8 -> 16x16
    upsample_cpu(dec_out1.data(), dec_up1.data(), BATCH, 8, 8, 128);

    // 4. Decoder L2
    // Conv: 16x16 -> 16x16
    ConvParam p4 = {BATCH, 16, 16, 128, 16, 16, 256, 3, 1, 1};
    conv2d_cpu(dec_up1.data(), dec_w2.data(), dec_b2.data(), dec_out2.data(), p4);
    relu_cpu(dec_out2.data(), dec_out2.size());
    // Upsample: 16x16 -> 32x32
    upsample_cpu(dec_out2.data(), dec_up2.data(), BATCH, 16, 16, 256);

    // 5. Output Layer (Conv về 3 channels)
    ConvParam p5 = {BATCH, 32, 32, 256, 32, 32, 3, 3, 1, 1};
    conv2d_cpu(dec_up2.data(), dec_w3.data(), dec_b3.data(), final_output.data(), p5);
    // Không dùng ReLU ở layer cuối cùng để reconstruction có thể linh hoạt

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // --- REPORT ---
    std::cout << "\n--- RESULTS ---\n";
    std::cout << "Total Forward Time: " << elapsed.count() << " s\n";
    std::cout << "Time per image: " << elapsed.count() / BATCH << " s\n";

    // Tính MSE Loss (So sánh Input gốc và Output tái tạo)
    float loss = mse_loss_cpu(final_output.data(), input_batch.data(), final_output.size());
    std::cout << "Reconstruction MSE Loss (Initial): " << loss << "\n";
    std::cout << "(Note: Loss is high because weights are random, not trained)\n";

    return 0;
}