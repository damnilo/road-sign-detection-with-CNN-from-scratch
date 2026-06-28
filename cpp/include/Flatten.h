#pragma once
#include "Layer.h"
#include <vector>

// Reshapes a multi-dimensional tensor (e.g. (N, C, H, W) from a conv stack)
// into a flat 2D tensor (N, C*H*W) suitable for feeding into a Dense layer.
class Flatten : public Layer
{
private:
    std::vector<size_t> inputShapeCache; // Original shape, needed to reshape gradients back in backward()
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};