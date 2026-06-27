#include "MaxPool2D.h"
# include <limits>

MaxPool2D::MaxPool2D(size_t pool_size, size_t stride)
    : pool_size(pool_size), stride(stride) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    inputShapeCache = input.getShape(); // Cache the original shape for backward pass

    size_t batch_size = input.getShape()[0];
    size_t channels = input.getShape()[1];
    size_t height = input.getShape()[2];
    size_t width = input.getShape()[3];

    size_t out_height = (height - pool_size) / stride + 1;
    size_t out_width = (width - pool_size) / stride + 1;

    Tensor output({batch_size, channels, out_height, out_width});
    maxIndicesCache.assign(output.size(), 0); // Initialize max indices cach
    
    const std::vector<float>& inputData = input.raw();
    size_t outputIndex = 0;

    for(size_t b = 0; b < batch_size; ++b){
        for(size_t c = 0; c < channels; ++c){
            for(size_t i = 0; i < out_height; ++i){
                size_t h_start = i * stride;
                for(size_t j = 0; j < out_width; ++j){
                    size_t w_start = j * stride;
                    float maxVal = std::numeric_limits<float>::lowest();
                    size_t maxIndex = 0;

                    for(size_t m = 0; m < pool_size; ++m){
                        size_t h_index = h_start + m;
                        for(size_t n = 0; n < pool_size; ++n){
                            size_t w_index = w_start + n;
                            size_t flatIndex = b * channels * height * width + c * height * width + h_index * width + w_index;
                            float val = input.raw()[flatIndex];
                            if(val > maxVal){
                                maxVal = val;
                                maxIndex = flatIndex;
                            }
                        }
                    }

                    output[outputIndex] = maxVal;
                    maxIndicesCache[outputIndex] = maxIndex;
                    ++outputIndex;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward(const Tensor& gradOutput) {
    Tensor gradInput = Tensor::zeros(inputShapeCache); // Initialize gradient input tensor with zeros
    for(size_t i = 0; i < gradOutput.size(); ++i) {
        size_t maxIndex = maxIndicesCache[i];
        gradInput.raw()[maxIndex] += gradOutput.raw()[i]; // Accumulate gradients
    }

    return gradInput;
}
