#include "Flatten.h"

// Collapses all dimensions after the batch axis into one, preserving element order
// (since the underlying buffer is already row-major / contiguous, this is just a copy).
Tensor Flatten::forward(const Tensor& input) {
    inputShapeCache = input.getShape();
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

// Reshapes the incoming flat gradient back to the original pre-flatten shape.
// No data movement needed since the buffer layout is unchanged, just shape metadata.
Tensor Flatten::backward(const Tensor& gradOutput) {
    Tensor gradInput = gradOutput;
    gradInput.reshape(inputShapeCache);
    return gradInput;
}