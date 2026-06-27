#include "Conv2D.h"
#include <cmath>

Tensor Conv2D::im2col(const Tensor& input, size_t kernel_size, size_t stride, size_t padding) {
    size_t batch_size = input.getShape()[0];
    size_t in_channels = input.getShape()[1];
    size_t in_height = input.getShape()[2];
    size_t in_width = input.getShape()[3];

    size_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    size_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    Tensor col({batch_size, in_channels * kernel_size * kernel_size, out_height * out_width});
    const std::vector<float>& inputData = input.raw();
    std::vector<float>& colData = col.raw();
    size_t rowSize = in_channels * kernel_size * kernel_size;
    size_t row = 0;

    for(size_t b = 0; b < batch_size; ++b){
        for(size_t c = 0; c < out_width; ++c){
            for(size_t i = 0; i < out_height; ++i){
                size_t col_idx = 0;
                long hStart = static_cast<long>(i * stride) - static_cast<long>(padding);
                long wStart = static_cast<long>(c * stride) - static_cast<long>(padding);
                for(size_t j = 0; j < in_channels; ++j){
                    for(size_t m = 0; m < kernel_size; ++m){
                        long h = hStart + m;
                        for(size_t n = 0; n < kernel_size; ++n){
                            long w = wStart + n;
                            float value = 0.0f;
                            if(h >= 0 && h < static_cast<long>(in_height) && w >= 0 && w < static_cast<long>(in_width)){
                                size_t inputIndex = b * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w;
                                value = inputData[inputIndex];
                            }
                            colData[row * rowSize + col_idx] = value;
                            ++col_idx;
                        }
                    }
                }
                ++row;
            }
        }
    }

    return col;
}

Tensor Conv2D::col2im(const Tensor& col, const std::vector<size_t>& input, size_t kernel_size, size_t stride, size_t padding) {
    size_t batch_size = input[0];
    size_t in_channels = input[1];
    size_t in_height = input[2];
    size_t in_width = input[3];

    size_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    size_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    Tensor gradInput = Tensor::zeros(input);
    std::vector<float>& gi = gradInput.raw();
    const std::vector<float>& colData = col.raw();
    size_t rowSize = in_channels * kernel_size * kernel_size;

    size_t row = 0;
    for(size_t b = 0; b < batch_size; ++b){
        for(size_t c = 0; c < out_width; ++c){
            for(size_t i = 0; i < out_height; ++i){
                size_t col_idx = 0;
                long hStart = static_cast<long>(i * stride) - static_cast<long>(padding);
                long wStart = static_cast<long>(c * stride) - static_cast<long>(padding);
                for(size_t j = 0; j < in_channels; ++j){
                    for(size_t m = 0; m < kernel_size; ++m){
                        long h = hStart + m;
                        for(size_t n = 0; n < kernel_size; ++n){
                            long w = wStart + n;
                            if(h >= 0 && h < static_cast<long>(in_height) && w >= 0 && w < static_cast<long>(in_width)){
                                size_t outputIndex = b * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w;
                                gi[outputIndex] += colData[row * rowSize + col_idx];
                            }
                            ++col_idx;
                        }
                    }
                }
                ++row;
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
    inputCache = input; // Cache the input for backward pass
    Tensor colCache = im2col(input, kernel_size, stride, padding);
    
    Tensor outCol = weights.matmul(colCache).broadcastAdd(biases);

    size_t N = inputCache.getShape()[0];
    Tensor output({N, out_channels, outHCache, outWCache});
    const std::vector<float>& src = outCol.raw();
    std::vector<float>& dst = output.raw();

    size_t row = 0;

    for(size_t n = 0; n < N; ++n){
        for(size_t c = 0; c < out_channels; ++c){
            for(size_t i = 0; i < outHCache; ++i){
                for(size_t j = 0; j < outWCache; ++j){
                    size_t dstIndex = ((n * out_channels + c) * outHCache + i) * outWCache + j;
                    dst[dstIndex] = src[row * out_channels + c];
                }
                ++row;
            }
        }
    }
    
    return output;
}

Tensor Conv2D::backward(const Tensor& gradOutput) {
    size_t N = inputCache.getShape()[0];
    Tensor dOutCol({N * outHCache * outWCache, out_channels});
    const std::vector<float>& src = gradOutput.raw();
    std::vector<float>& dst = dOutCol.raw();

    size_t row = 0;
    for(size_t n = 0; n < N; ++n){
        for(size_t c = 0; c < out_channels; ++c){
            for(size_t i = 0; i < outHCache; ++i){
                for(size_t j = 0; j < outWCache; ++j){
                    size_t srcIndex = ((n * out_channels + c) * outHCache + i) * outWCache + j;
                    dst[row * out_channels + c] = src[srcIndex];
                }
                ++row;
            }
        }
    }

    gradWeights = dOutCol.transpose().matmul(dOutCol);
    gradBiases = dOutCol.sum(0);

    Tensor dCol = dOutCol.matmul(weights.transpose());

    return col2im(dCol, inputCache.getShape(), kernel_size, stride, padding);
}

std::vector<Tensor*> Conv2D::parameters() {
    return {&weights, &biases};
}

std::vector<Tensor*> Conv2D::gradients() {
    return {&gradWeights, &gradBiases};
}