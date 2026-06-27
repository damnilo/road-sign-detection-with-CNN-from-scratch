#pragma once
#include "Layer.h"
#include <vector>

class Flatten : public Layer
{
private:
    std::vector<size_t> inputShapeCache; // To store the original shape for backward pass
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};