#include "cifar10_dataset.h"

CIFAR10Dataset::CIFAR10Dataset(const std::string& path) : data_dir(path) {
    // Dự trù bộ nhớ để tránh cấp phát lại nhiều lần
    // 50,000 ảnh * 3072 float ~ 600MB RAM
    train_images.reserve(50000 * IMG_SIZE);
    train_labels.reserve(50000);
    
    test_images.reserve(10000 * IMG_SIZE);
    test_labels.reserve(10000);
}

void CIFAR10Dataset::read_batch(const std::string& filename, std::vector<float>& images, std::vector<unsigned char>& labels) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return; // Hoặc throw exception tùy ý
    }

    // Buffer tạm để đọc 1 record (1 byte label + 3072 bytes pixel)
    std::vector<unsigned char> buffer(RECORD_SIZE);

    // Đọc liên tục cho đến hết file (mỗi file batch thường có 10,000 ảnh)
    while (file.read((char*)buffer.data(), RECORD_SIZE)) {
        // 1. Xử lý Label (Byte đầu tiên)
        labels.push_back(buffer[0]);

        // 2. Xử lý Image (3072 bytes tiếp theo)
        // CIFAR-10 binary format: [Label] [1024 Red] [1024 Green] [1024 Blue]
        // Chuẩn hóa từ [0, 255] sang [0.0f, 1.0f] [cite: 140, 240]
        for (int i = 0; i < IMG_SIZE; ++i) {
            images.push_back(static_cast<float>(buffer[i + 1]) / 255.0f);
        }
    }
    std::cout << "Loaded batch: " << filename << " | Current Total: " << labels.size() << std::endl;
}

void CIFAR10Dataset::load_data() {
    std::cout << "--- Loading CIFAR-10 Dataset ---" << std::endl;

    // Load 5 file training (data_batch_1.bin -> data_batch_5.bin) [cite: 33, 44]
    for (int i = 1; i <= 5; ++i) {
        std::string filename = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        read_batch(filename, train_images, train_labels);
    }

    // Load 1 file test (test_batch.bin) [cite: 34, 46]
    std::string test_filename = data_dir + "/test_batch.bin";
    read_batch(test_filename, test_images, test_labels);

    std::cout << "Successfully loaded " << get_num_train() << " train images and " 
              << get_num_test() << " test images." << std::endl;
}