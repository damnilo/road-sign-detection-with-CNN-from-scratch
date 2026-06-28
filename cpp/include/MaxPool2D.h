#pragma once
#include "Layer.h"
#include <vector>

// 2D max pooling: downsamples each (pool_size x pool_size) window to its max value,
// independently per channel and batch item.
class MaxPool2D : public Layer
{
private:
    size_t pool_size;
    size_t stride;

    std::vector<size_t> inputShapeCache; // Original (N,C,H,W) shape, needed to size backward()'s output
    std::vector<size_t> maxIndicesCache; // Flat input-index of the max element for each output position

public:
    MaxPool2D(size_t pool_size, size_t stride);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};