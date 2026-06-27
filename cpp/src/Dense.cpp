#include "Dense.h"
#include <cmath>

Dense::Dense(size_t inputSize, size_t outputSize)
    : inputSize(inputSize), outputSize(outputSize),
      weights(Tensor::zeros({outputSize, inputSize})),
      biases(Tensor::zeros({outputSize})),
      gradWeights(Tensor::zeros({outputSize, inputSize})),
      gradBiases(Tensor::zeros({outputSize}))
{
    float limit = std::sqrt(6.0f / (inputSize + outputSize));
    weights.randomize(-limit, limit);
    biases.fill(0.0f);
}

Tensor Dense::forward(const Tensor& input) {
    inputCache = input; // Cache the input for backward pass

    size_t batch = input.getShape()[0];
    size_t features = input.getShape()[1];

    if(features != inputSize){
        throw std::invalid_argument("Input size does not match layer's input size");
    }

    Tensor output({batch, outputSize});
    output = input.matmul(weights.transpose()).broadcastAdd(biases);
    return output;
}

Tensor Dense::backward(const Tensor& gradOutput) {
    if(gradOutput.getShape()[0] != outputSize){
        throw std::invalid_argument("Gradient output size does not match layer's output size");
    }

    // Compute gradients
    gradWeights = gradOutput.matmul(inputCache.transpose());
    gradBiases = gradOutput;

    // Compute gradient with respect to input
    Tensor gradInput = weights.transpose().matmul(gradOutput);
    return gradInput;
}

std::vector<Tensor*> Dense::parameters() {
    return {&weights, &biases};
}

std::vector<Tensor*> Dense::gradients() {
    return {&gradWeights, &gradBiases};
}
