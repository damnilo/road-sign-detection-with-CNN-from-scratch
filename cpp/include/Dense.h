#pragma once
#include "Layer.h"

class Dense : public Layer
{
private:
    size_t inputSize;
    size_t outputSize;

    Tensor weights;
    Tensor biases;
    Tensor inputCache; // Cache the input for use in backward pass
    Tensor gradWeights;
    Tensor gradBiases;

public:
    Dense(size_t inputSize, size_t outputSize);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
};