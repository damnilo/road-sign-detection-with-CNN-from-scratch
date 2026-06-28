#pragma once
#include "Layer.h"
#include <vector>

// 2D convolution layer, implemented via the im2col/matmul/col2im trick: the
// convolution is reframed as a single large matrix multiply by unrolling input
// patches into rows, which lets it reuse Tensor::matmul instead of a naive
// 6-nested-loop convolution.
class Conv2D : public Layer
{
private:
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    Tensor weights;     // Shape (out_channels, in_channels * kernel_size * kernel_size)
    Tensor biases;      // Shape (out_channels,)
    Tensor inputCache;  // Raw input from forward(), needed for col2im in backward()
    Tensor colCache;    // im2col result from forward(), needed for gradWeights in backward()
    Tensor gradWeights;
    Tensor gradBiases;

    std::vector<size_t> inputShapeCache; // Original (N,C,H,W) shape, needed to size col2im's output
    size_t outHCache;
    size_t outWCache;

    // Unrolls each (kernel_size x kernel_size) receptive field across all channels into
    // one row. Output shape: (N*outH*outW, in_channels*kernel_size*kernel_size).
    static Tensor im2col(const Tensor& input, size_t kernel_size, size_t stride, size_t padding);

    // Inverse of im2col: scatters/accumulates column-format gradients back into the
    // original (N,C,H,W) spatial layout (overlapping receptive fields sum their gradients).
    static Tensor col2im(const Tensor& col, const std::vector<size_t>& input, size_t kernel_size, size_t stride, size_t padding);

public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
};