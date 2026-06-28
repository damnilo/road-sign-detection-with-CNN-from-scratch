#pragma once
#include "Layer.h"

// Fully-connected (linear) layer: output = input @ weights + biases.
// Expects 2D input of shape (N, inputSize); produces (N, outputSize).
class Dense : public Layer
{
private:
    size_t inputSize;
    size_t outputSize;

    Tensor weights;     // Shape (inputSize, outputSize)
    Tensor biases;      // Shape (outputSize,)
    Tensor inputCache;  // Input from forward(), needed to compute gradWeights in backward()
    Tensor gradWeights;
    Tensor gradBiases;

public:
    Dense(size_t inputSize, size_t outputSize);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
};