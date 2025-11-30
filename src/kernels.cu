#include "kernels.h"
#include <algorithm> // Cho std::fill

// Helper nội bộ (không cần export ra header)
inline int get_idx(int b, int h, int w, int c, int H, int W, int C) {
    return b * (H * W * C) + h * (W * C) + w * C + c;
}

// 1. Convolution
void conv2d(float* input, float* weight, float* bias, float* output, ConvParam p) {
    // ... (Giữ nguyên logic vòng lặp for cũ) ...
    for (int b = 0; b < p.batch; ++b) {
        for (int oh = 0; oh < p.out_h; ++oh) {
            for (int ow = 0; ow < p.out_w; ++ow) {
                for (int oc = 0; oc < p.out_c; ++oc) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < p.in_c; ++ic) {
                        for (int kh = 0; kh < p.k_size; ++kh) {
                            for (int kw = 0; kw < p.k_size; ++kw) {
                                int ih = oh * p.stride - p.padding + kh;
                                int iw = ow * p.stride - p.padding + kw;
                                if (ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w) {
                                    int in_idx = get_idx(b, ih, iw, ic, p.in_h, p.in_w, p.in_c);
                                    int w_idx = oc * (p.in_c * p.k_size * p.k_size) 
                                              + ic * (p.k_size * p.k_size) 
                                              + kh * p.k_size + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = get_idx(b, oh, ow, oc, p.out_h, p.out_w, p.out_c);
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// 2. ReLU
void relu(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        if (data[i] < 0) data[i] = 0.0f;
    }
}

// 3. Max Pooling
void maxpool(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int stride = 2;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -1e9;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            int in_idx = get_idx(b, ih, iw, c, in_h, in_w, in_c);
                            if (input[in_idx] > max_val) max_val = input[in_idx];
                        }
                    }
                    int out_idx = get_idx(b, oh, ow, c, out_h, out_w, in_c);
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

// 4. Upsample
void upsample(float* input, float* output, int batch, int in_h, int in_w, int in_c) {
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / 2;
                    int iw = ow / 2;
                    int in_idx = get_idx(b, ih, iw, c, in_h, in_w, in_c);
                    int out_idx = get_idx(b, oh, ow, c, out_h, out_w, in_c);
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// 5. MSE Loss
float mse_loss(float* pred, float* target, int size) {
    float sum = 0.0f;
    for(int i=0; i<size; ++i) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum/size;
}

// --- BACKWARD IMPLEMENTATIONS ---

void conv2d_backward(float* d_output, float* input, float* weight, 
                     float* d_input, float* d_weight, float* d_bias, ConvParam p) {
    std::fill(d_input, d_input + (p.batch * p.in_h * p.in_w * p.in_c), 0.0f);
    std::fill(d_weight, d_weight + (p.out_c * p.in_c * p.k_size * p.k_size), 0.0f);
    std::fill(d_bias, d_bias + p.out_c, 0.0f);

    for (int b = 0; b < p.batch; ++b) {
        for (int oh = 0; oh < p.out_h; ++oh) {
            for (int ow = 0; ow < p.out_w; ++ow) {
                for (int oc = 0; oc < p.out_c; ++oc) {
                    int out_idx = get_idx(b, oh, ow, oc, p.out_h, p.out_w, p.out_c);
                    float d_val = d_output[out_idx];
                    d_bias[oc] += d_val;

                    for (int ic = 0; ic < p.in_c; ++ic) {
                        for (int kh = 0; kh < p.k_size; ++kh) {
                            for (int kw = 0; kw < p.k_size; ++kw) {
                                int ih = oh * p.stride - p.padding + kh;
                                int iw = ow * p.stride - p.padding + kw;
                                if (ih >= 0 && ih < p.in_h && iw >= 0 && iw < p.in_w) {
                                    int in_idx = get_idx(b, ih, iw, ic, p.in_h, p.in_w, p.in_c);
                                    int w_idx = oc * (p.in_c * p.k_size * p.k_size) 
                                              + ic * (p.k_size * p.k_size) 
                                              + kh * p.k_size + kw;
                                    d_weight[w_idx] += input[in_idx] * d_val;
                                    d_input[in_idx] += d_val * weight[w_idx];
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
        d_input[i] = (input[i] > 0) ? d_output[i] : 0.0f;
    }
}

void maxpool_backward(float* d_output, float* input, float* d_input, 
                      int batch, int in_h, int in_w, int in_c) {
    std::fill(d_input, d_input + (batch * in_h * in_w * in_c), 0.0f);
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int stride = 2;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int max_idx = -1;
                    float max_val = -1e9;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            int in_idx = get_idx(b, ih, iw, c, in_h, in_w, in_c);
                            if (input[in_idx] > max_val) {
                                max_val = input[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    int out_idx = get_idx(b, oh, ow, c, out_h, out_w, in_c);
                    if (max_idx != -1) d_input[max_idx] += d_output[out_idx];
                }
            }
        }
    }
}

void upsample_backward(float* d_output, float* d_input, int batch, int in_h, int in_w, int in_c) {
    std::fill(d_input, d_input + (batch * in_h * in_w * in_c), 0.0f);
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / 2;
                    int iw = ow / 2;
                    int in_idx = get_idx(b, ih, iw, c, in_h, in_w, in_c);
                    int out_idx = get_idx(b, oh, ow, c, out_h, out_w, in_c);
                    d_input[in_idx] += d_output[out_idx];
                }
            }
        }
    }
}

void update_weights(float* weights, float* d_weights, int size, float lr) {
    for (int i = 0; i < size; ++i) {
        weights[i] = weights[i] - lr * d_weights[i];
    }
}
