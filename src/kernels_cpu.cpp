// #include "kernel.h"
#include "kernel_cpu.h"
#include <cstring> // Cho memset
#include <algorithm> // Cho std::max
#include <cmath>     // Cho isinf, etc.

// Macro hỗ trợ lấy index cho định dạng NCHW (Batch, Channel, Height, Width)
#define IDX_NCHW(n, c, h, w, C, H, W) ((n) * (C) * (H) * (W) + (c) * (H) * (W) + (h) * (W) + (w))
// Macro lấy index cho Weight (OutChannel, InChannel, KernelH, KernelW)
#define IDX_W(oc, ic, kh, kw, Cin, K) ((oc) * (Cin) * (K) * (K) + (ic) * (K) * (K) + (kh) * (K) + (kw))

// ==========================================
// FORWARD PASS
// ==========================================

void conv2d(float* input, float* weight, float* bias, float* output, ConvParam p) {
    // Kích thước output đã được tính toán bên ngoài và truyền vào p.out_h, p.out_w
    // Logic: Duyệt qua từng pixel output và tính tổng chập
    
    for (int b = 0; b < p.batch; ++b) {
        for (int oc = 0; oc < p.out_c; ++oc) {
            for (int oh = 0; oh < p.out_h; ++oh) {
                for (int ow = 0; ow < p.out_w; ++ow) {
                    
                    // Khởi tạo giá trị bằng Bias (nếu có)
                    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
                    
                    // Convolution sum
                    for (int ic = 0; ic < p.in_c; ++ic) {
                        for (int kh = 0; kh < p.k_size; ++kh) {
                            for (int kw = 0; kw < p.k_size; ++kw) {
                                // Tính chỉ số input tương ứng
                                int ih = oh * p.stride - p.padding + kh;
                                int iw = ow * p.stride - p.padding + kw;
                                
                                // Boundary check (Padding)
                                if (ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w) {
                                    int in_idx = IDX_NCHW(b, ic, ih, iw, p.in_c, p.in_h, p.in_w);
                                    int w_idx  = IDX_W(oc, ic, kh, kw, p.in_c, p.k_size);
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    
                    int out_idx = IDX_NCHW(b, oc, oh, ow, p.out_c, p.out_h, p.out_w);
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void relu(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// Giả định MaxPool 2x2, stride 2 (tiêu chuẩn cho CIFAR-10 autoencoder)
void maxpool(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    
                    float max_val = -1e9f; // Số rất nhỏ
                    
                    // Duyệt vùng 2x2
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int ih = oh * 2 + kh;
                            int iw = ow * 2 + kw;
                            
                            int idx = IDX_NCHW(b, c, ih, iw, in_c, in_h, in_w);
                            if (input[idx] > max_val) {
                                max_val = input[idx];
                            }
                        }
                    }
                    
                    int out_idx = IDX_NCHW(b, c, oh, ow, in_c, out_h, out_w);
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

// Upsample 2x2 (Nearest Neighbor)
void upsample(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    // Map pixel đầu ra về đầu vào (chia 2 lấy phần nguyên)
                    int ih = h / 2;
                    int iw = w / 2;
                    
                    int in_idx = IDX_NCHW(b, c, ih, iw, in_c, in_h, in_w);
                    int out_idx = IDX_NCHW(b, c, h, w, in_c, out_h, out_w);
                    
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

float mse_loss(float* pred, float* target, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum / size;
}

// ==========================================
// BACKWARD PASS
// ==========================================

void conv2d_backward(float* d_output, float* input, float* weight, 
                     float* d_input, float* d_weight, float* d_bias, ConvParam p) {
    
    // 1. Reset Gradients về 0 (quan trọng vì ta dùng toán tử +=)
    if (d_input) std::memset(d_input, 0, p.batch * p.in_c * p.in_h * p.in_w * sizeof(float));
    if (d_weight) std::memset(d_weight, 0, p.out_c * p.in_c * p.k_size * p.k_size * sizeof(float));
    if (d_bias) std::memset(d_bias, 0, p.out_c * sizeof(float));

    // 2. Loop để tích lũy gradient
    for (int b = 0; b < p.batch; ++b) {
        for (int oc = 0; oc < p.out_c; ++oc) {
            for (int oh = 0; oh < p.out_h; ++oh) {
                for (int ow = 0; ow < p.out_w; ++ow) {
                    
                    // Gradient từ layer sau truyền về
                    int dout_idx = IDX_NCHW(b, oc, oh, ow, p.out_c, p.out_h, p.out_w);
                    float dout_val = d_output[dout_idx];

                    // a. Gradient cho Bias: Tổng d_output
                    if (d_bias) {
                        d_bias[oc] += dout_val;
                    }

                    // b. Gradient cho Weights và Input
                    for (int ic = 0; ic < p.in_c; ++ic) {
                        for (int kh = 0; kh < p.k_size; ++kh) {
                            for (int kw = 0; kw < p.k_size; ++kw) {
                                int ih = oh * p.stride - p.padding + kh;
                                int iw = ow * p.stride - p.padding + kw;

                                if (ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w) {
                                    // Calc d_weight
                                    if (d_weight) {
                                        int in_idx = IDX_NCHW(b, ic, ih, iw, p.in_c, p.in_h, p.in_w);
                                        int w_idx = IDX_W(oc, ic, kh, kw, p.in_c, p.k_size);
                                        
                                        // dW += Input * dOut
                                        d_weight[w_idx] += input[in_idx] * dout_val;
                                    }

                                    // Calc d_input
                                    if (d_input) {
                                        int in_idx = IDX_NCHW(b, ic, ih, iw, p.in_c, p.in_h, p.in_w);
                                        int w_idx = IDX_W(oc, ic, kh, kw, p.in_c, p.k_size);

                                        // dX += Weight * dOut
                                        d_input[in_idx] += weight[w_idx] * dout_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void relu_backward(float* d_output, float* input, float* d_input, int size) {
    for (int i = 0; i < size; ++i) {
        // Nếu input forward > 0 thì gradient truyền qua, ngược lại bằng 0
        d_input[i] = (input[i] > 0.0f) ? d_output[i] : 0.0f;
    }
}

void maxpool_backward(float* d_output, float* input, float* d_input, 
                      int batch, int in_h, int in_w, int in_c) {
    // Reset d_input
    std::memset(d_input, 0, batch * in_c * in_h * in_w * sizeof(float));

    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    
                    // Tìm lại vị trí Max trong forward pass
                    // (Vì header không lưu indices, ta phải tìm lại)
                    float max_val = -1e9f;
                    int max_idx = -1;

                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int ih = oh * 2 + kh;
                            int iw = ow * 2 + kw;
                            int idx = IDX_NCHW(b, c, ih, iw, in_c, in_h, in_w);
                            
                            if (input[idx] > max_val) {
                                max_val = input[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    
                    // Truyền gradient về đúng vị trí max đó
                    int dout_idx = IDX_NCHW(b, c, oh, ow, in_c, out_h, out_w);
                    if (max_idx != -1) {
                        d_input[max_idx] += d_output[dout_idx];
                    }
                }
            }
        }
    }
}

void upsample_backward(float* d_output, float* d_input, int batch, int in_h, int in_w, int in_c) {
    // Reset d_input
    // Lưu ý: in_h, in_w ở đây là kích thước CỦA LAYER TRƯỚC (nhỏ), 
    // d_output là kích thước LỚN (đã upsample).
    // Nhưng tham số hàm tên là in_h, in_w ứng với d_input.
    std::memset(d_input, 0, batch * in_c * in_h * in_w * sizeof(float));
    
    int out_h = in_h * 2;
    int out_w = in_w * 2;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    // Pixel lớn (h,w) được map từ pixel nhỏ (h/2, w/2)
                    int ih = h / 2;
                    int iw = w / 2;
                    
                    int in_idx = IDX_NCHW(b, c, ih, iw, in_c, in_h, in_w);
                    int out_idx = IDX_NCHW(b, c, h, w, in_c, out_h, out_w);
                    
                    // Cộng dồn gradient
                    d_input[in_idx] += d_output[out_idx];
                }
            }
        }
    }
}

void update_weights(float* weights, float* d_weights, int size, float lr) {
    for (int i = 0; i < size; ++i) {
        weights[i] -= lr * d_weights[i];
    }
}