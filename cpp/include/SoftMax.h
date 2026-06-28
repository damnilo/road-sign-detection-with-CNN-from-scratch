#pragma once
#include "Layer.h"

// Softmax activation, applied row-wise over a 2D (N, C) tensor of class logits.
// Typically the final layer before CrossEntropy loss.
class SoftMax : public Layer
{
private:
    Tensor outputCache; // Softmax probabilities from forward(), reused in backward()

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};