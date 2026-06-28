#include "ReLU.h"

// Stores a binary mask alongside the clamped output so backward() can route
// gradients only through positions that were active during the forward pass.
Tensor ReLU::forward(const Tensor& input) {
    maskCache = Tensor(input.getShape());
    Tensor output(input.getShape());
    for (size_t i = 0; i < input.size(); ++i) {
        float v = input[i];
        bool isPositive = v > 0;
        output[i] = isPositive ? v : 0.0f;
        maskCache[i] = isPositive ? 1.0f : 0.0f;
    }
    return output;
}

// dL/dInput = dL/dOutput * mask (zero out gradients wherever the forward input was <= 0)
Tensor ReLU::backward(const Tensor& gradOutput) {
    return gradOutput * maskCache;
}