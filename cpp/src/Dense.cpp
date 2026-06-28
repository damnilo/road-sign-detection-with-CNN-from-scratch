#include "Dense.h"
#include <cmath>
#include <iostream>

// He-style fan-in/fan-out uniform initialization, scaled for ReLU-friendly variance.
// Biases start at zero, which is standard.
Dense::Dense(size_t inputSize, size_t outputSize)
    : inputSize(inputSize), outputSize(outputSize),
      weights({inputSize, outputSize}),
      biases({outputSize}),
      gradWeights({inputSize, outputSize}),
      gradBiases({outputSize})
{
    float limit = std::sqrt(6.0f / (inputSize + outputSize));
    weights.randomize(-limit, limit);
    biases.fill(0.0f);
}

// output = input @ weights + biases (biases broadcast across the batch axis)
Tensor Dense::forward(const Tensor& input)
{
    inputCache = input; // Needed in backward() to compute gradWeights

    return input.matmul(weights)
               .broadcastAdd(biases);
}

// Standard linear-layer backward pass:
//   gradWeights = input^T @ gradOutput
//   gradBiases  = sum(gradOutput) over the batch axis
//   gradInput   = gradOutput @ weights^T
Tensor Dense::backward(const Tensor& gradOutput) {
    gradWeights = inputCache.transpose().matmul(gradOutput);
    gradBiases = gradOutput.sum(0);

    Tensor gradInput = gradOutput.matmul(weights.transpose());

    return gradInput;
}

std::vector<Tensor*> Dense::parameters() {
    return {&weights, &biases};
}

std::vector<Tensor*> Dense::gradients() {
    return {&gradWeights, &gradBiases};
}