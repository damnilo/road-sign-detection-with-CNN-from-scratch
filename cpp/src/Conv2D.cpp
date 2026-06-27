#pragma omp parallel for
#include "Conv2D.h"
#include <cmath>

Tensor Conv2D::im2col(const Tensor& input, size_t kernel_size, size_t stride, size_t padding) {
    size_t batch_size = input.getShape()[0];
    size_t in_channels = input.getShape()[1];
    size_t in_height = input.getShape()[2];
    size_t in_width = input.getShape()[3];

    size_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    size_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    size_t rowSize = in_channels * kernel_size * kernel_size;

    Tensor col({batch_size * out_height * out_width, rowSize});
    const std::vector<float>& inputData = input.raw();
    std::vector<float>& colData = col.raw();

    #pragma omp parallel for
    for(size_t b = 0; b < batch_size; ++b){
        for(size_t i = 0; i < out_height; ++i){
            long hStart = static_cast<long>(i * stride) - static_cast<long>(padding);
            for(size_t w = 0; w < out_width; ++w){
                long wStart = static_cast<long>(w * stride) - static_cast<long>(padding);
                size_t row = (b * out_height + i) * out_width + w;
                size_t col_idx = 0;

                for(size_t ch = 0; ch < in_channels; ++ch){
                    for(size_t m = 0; m < kernel_size; ++m){
                        long h = hStart + m;
                        for(size_t n = 0; n < kernel_size; ++n){
                            long wIdx = wStart + n;
                            float value = 0.0f;
                            if(h >= 0 && h < static_cast<long>(in_height) && wIdx >= 0 && wIdx < static_cast<long>(in_width)){
                                size_t inputIndex = ((b * in_channels + ch) * in_height + h) * in_width + wIdx;
                                value = inputData[inputIndex];
                            }
                            colData[row * rowSize + col_idx] = value;
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }

    return col;
}

Tensor Conv2D::col2im(const Tensor& col, const std::vector<size_t>& inputShape, size_t kernel_size, size_t stride, size_t padding) {
    size_t batch_size = inputShape[0];
    size_t in_channels = inputShape[1];
    size_t in_height = inputShape[2];
    size_t in_width = inputShape[3];

    size_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    size_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    size_t rowSize = in_channels * kernel_size * kernel_size;

    Tensor gradInput = Tensor::zeros(inputShape);
    std::vector<float>& gi = gradInput.raw();
    const std::vector<float>& colData = col.raw();

    #pragma omp parallel for
    for(size_t b = 0; b < batch_size; ++b){
        for(size_t i = 0; i < out_height; ++i){
            long hStart = static_cast<long>(i * stride) - static_cast<long>(padding);
            for(size_t w = 0; w < out_width; ++w){
                long wStart = static_cast<long>(w * stride) - static_cast<long>(padding);
                size_t row = (b * out_height + i) * out_width + w;
                size_t col_idx = 0;

                for(size_t ch = 0; ch < in_channels; ++ch){
                    for(size_t m = 0; m < kernel_size; ++m){
                        long h = hStart + m;
                        for(size_t n = 0; n < kernel_size; ++n){
                            long wIdx = wStart + n;
                            if(h >= 0 && h < static_cast<long>(in_height) && wIdx >= 0 && wIdx < static_cast<long>(in_width)){
                                size_t inputIndex = ((b * in_channels + ch) * in_height + h) * in_width + wIdx;
                                gi[inputIndex] += colData[row * rowSize + col_idx];
                            }
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }

    return gradInput;
}

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding),
      weights(Tensor::zeros({out_channels, in_channels * kernel_size * kernel_size})),
      biases(Tensor::zeros({out_channels})),
      gradWeights(Tensor::zeros({out_channels, in_channels * kernel_size * kernel_size})),
      gradBiases(Tensor::zeros({out_channels})) {
    size_t fanIn = in_channels * kernel_size * kernel_size;
    float limit = std::sqrt(6.0f / (fanIn + out_channels));
    weights.randomize(-limit, limit);
    biases.fill(0.0f);
}

Tensor Conv2D::forward(const Tensor& input) {
    inputCache = input;

    size_t N = input.getShape()[0];
    size_t inH = input.getShape()[2];
    size_t inW = input.getShape()[3];
    outHCache = (inH + 2 * padding - kernel_size) / stride + 1;
    outWCache = (inW + 2 * padding - kernel_size) / stride + 1;

    this->colCache = im2col(input, kernel_size, stride, padding);              // (N*outH*outW, rowSize)
    Tensor outCol = colCache.matmul(weights.transpose()).broadcastAdd(biases);  // (N*outH*outW, out_channels)

    Tensor output({N, out_channels, outHCache, outWCache});
    std::vector<float>& dst = output.raw();
    const auto& outData = outCol.raw();

    #pragma omp parallel for
    for(size_t b = 0; b < N; ++b){
        for(size_t i = 0; i < outHCache; ++i){
            for(size_t j = 0; j < outWCache; ++j){
                size_t row = (b * outHCache + i) * outWCache + j;
                for(size_t c = 0; c < out_channels; ++c){
                    size_t dstIndex = ((b * out_channels + c) * outHCache + i) * outWCache + j;
                    dst[dstIndex] = outData[row * out_channels + c];
                }
            }
        }
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& gradOutput) {
    size_t N = inputCache.getShape()[0];

    Tensor dOutCol({N * outHCache * outWCache, out_channels});
    const std::vector<float>& src = gradOutput.raw();
    auto& dOutData = dOutCol.raw();

    #pragma omp parallel for
    for(size_t b = 0; b < N; ++b){
        for(size_t i = 0; i < outHCache; ++i){
            for(size_t j = 0; j < outWCache; ++j){
                size_t row = (b * outHCache + i) * outWCache + j;
                for(size_t c = 0; c < out_channels; ++c){
                    size_t srcIndex = ((b * out_channels + c) * outHCache + i) * outWCache + j;

                    dOutData[row * out_channels + c] = src[srcIndex];
                }
            }
        }
    }

    gradWeights = dOutCol.transpose().matmul(colCache); // (out_channels, rowSize)
    gradBiases = dOutCol.sum(0);                          // (out_channels)

    Tensor dCol = dOutCol.matmul(weights);                // (N*outH*outW, rowSize)  -- note: no transpose, this was the other bug

    return col2im(dCol, inputCache.getShape(), kernel_size, stride, padding);
}

std::vector<Tensor*> Conv2D::parameters() {
    return {&weights, &biases};
}

std::vector<Tensor*> Conv2D::gradients() {
    return {&gradWeights, &gradBiases};
}