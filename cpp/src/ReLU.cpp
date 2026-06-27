#include "ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    maskCache = Tensor(input.getShape()); // Cache the input for backward pass
    Tensor output(input.getShape()); // Create a copy of the input tensor
    for (size_t i = 0; i < input.size(); ++i) {
        float v = input[i];
        bool isPositive = v > 0;
        output[i] = isPositive ? v : 0.0f; // Apply Re
        maskCache[i] = isPositive ? 1.0f : 0.0f; // Store the mask
    }
    return output;
}

Tensor ReLU::backward(const Tensor& gradOutput) {
    return gradOutput * maskCache; 
}