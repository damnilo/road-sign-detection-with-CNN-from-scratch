#pragma once
#include "Layer.h"

// Elementwise ReLU activation: output = max(0, input). No learnable parameters.
class ReLU : public Layer
{
private:
    Tensor maskCache; // 1.0 where input was positive, 0.0 otherwise — reused in backward()

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};