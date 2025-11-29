#include <iostream>
#include "cifar10_dataset.h"

int main() {
    // Giả sử folder chứa các file .bin nằm ở "./cifar-10-batches-bin"
    // Bạn cần tải dataset về và giải nén trước khi chạy
    std::string dataset_path = "../data/cifar-10-batches-bin"; 

    CIFAR10Dataset dataset(dataset_path);
    dataset.load_data();

    // KIỂM TRA DỮ LIỆU
    if (dataset.get_num_train() > 0) {
        // Lấy con trỏ dữ liệu (Dạng float* phẳng -> Sẵn sàng cho CUDA)
        float* d_train = dataset.get_train_images_ptr();
        unsigned char* l_train = dataset.get_train_labels_ptr();

        std::cout << "\nSample Data Check (Image 0):\n";
        std::cout << "Label: " << (int)l_train[0] << std::endl;
        std::cout << "First 5 pixels (Normalized): ";
        for(int i = 0; i < 5; ++i) {
            std::cout << d_train[i] << " ";
        }
        std::cout << std::endl;
        
        // Kiểm tra kích thước bộ nhớ
        // 50000 ảnh * 3072 float * 4 bytes = ~614 MB
        size_t mem_size = dataset.train_images.size() * sizeof(float);
        std::cout << "Total Training Memory Size: " << mem_size / (1024*1024) << " MB" << std::endl;
    } else {
        std::cout << "Failed to load data. Please check the path." << std::endl;
    }

    return 0;
}