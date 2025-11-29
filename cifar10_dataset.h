#ifndef CIFAR10_DATASET_H
#define CIFAR10_DATASET_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

// Cấu hình Dataset [cite: 30-36]
const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3;
const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS; // 3072
const int LABEL_SIZE = 1;
const int RECORD_SIZE = LABEL_SIZE + IMG_SIZE; // 3073 bytes

class CIFAR10Dataset {
private:
    std::string data_dir;
    
    // Hàm helper để đọc một file binary
    void read_batch(const std::string& filename, std::vector<float>& images, std::vector<unsigned char>& labels);

public:
    // Buffer chứa dữ liệu (Flattened Arrays)
    // Train: 50,000 * 3072 floats
    std::vector<float> train_images;
    std::vector<unsigned char> train_labels;

    // Test: 10,000 * 3072 floats
    std::vector<float> test_images;
    std::vector<unsigned char> test_labels;

    // Constructor: Nhận đường dẫn thư mục chứa các file data_batch_*.bin
    CIFAR10Dataset(const std::string& path);

    // Hàm load dữ liệu [cite: 137-140]
    void load_data();

    // Getter trả về con trỏ thô (Raw Pointer) để dùng cho CUDA sau này
    float* get_train_images_ptr() { return train_images.data(); }
    unsigned char* get_train_labels_ptr() { return train_labels.data(); }
    
    float* get_test_images_ptr() { return test_images.data(); }
    unsigned char* get_test_labels_ptr() { return test_labels.data(); }

    // Thông tin kích thước
    size_t get_num_train() const { return train_labels.size(); }
    size_t get_num_test() const { return test_labels.size(); }
};

#endif