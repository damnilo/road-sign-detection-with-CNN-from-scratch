#include "Flatten.h"

Tensor Flatten::forward(const Tensor& input) {
    inputShapeCache = input.getShape(); // Cache the original shape for backward pass
    size_t totalSize = inputShapeCache[0];
    size_t features = input.size() / totalSize;

    Tensor output({totalSize, features});

    const std::vector<float>& in = input.raw();
    std::vector<float>& dst = output.raw();
    
    for(size_t i = 0; i < totalSize; ++i){
        for(size_t j = 0; j < features; ++j){
            dst[i * features + j] = in[i * features + j];
        }
    }

    return output;
}

Tensor Flatten::backward(const Tensor& gradOutput) {
    Tensor gradInput = gradOutput;
    gradInput.reshape(inputShapeCache); // Reshape back to original shape
    return gradInput;
}