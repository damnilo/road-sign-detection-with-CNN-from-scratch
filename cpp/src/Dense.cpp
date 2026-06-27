#include "Dense.h"
#include <cmath>
#include <iostream>

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

Tensor Dense::forward(const Tensor& input)
{

    inputCache = input;

    return input.matmul(weights)
               .broadcastAdd(biases);
}

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
