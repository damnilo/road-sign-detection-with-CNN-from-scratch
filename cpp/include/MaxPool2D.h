#pragma once
#include "Layer.h"
#include <vector>

class MaxPool2D : public Layer
{
private:
    size_t pool_size;
    size_t stride;

    std::vector<size_t> inputShapeCache; // To store the original shape for backward pass
    std::vector<size_t> maxIndicesCache; // To store the indices of max values

public:
    MaxPool2D(size_t pool_size, size_t stride);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};