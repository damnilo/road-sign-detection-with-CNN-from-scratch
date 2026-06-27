#pragma once
#include "Layer.h"
#include <vector>

class Conv2D : public Layer
{
private:
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    Tensor weights;
    Tensor biases;
    Tensor inputCache; // Cache the input for use in backward pass
    Tensor gradWeights;
    Tensor gradBiases;

    std::vector<size_t> inputShapeCache; // To store the original shape for backward pass
    size_t outHCache; // To store the output height for backward pass
    size_t outWCache; // To store the output width for backward pass

    static Tensor im2col(const Tensor& input, size_t kernel_size, size_t stride, size_t padding);
    static Tensor col2im(const Tensor& col, const std::vector<size_t>& input, size_t kernel_size, size_t stride, size_t padding);

public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
};