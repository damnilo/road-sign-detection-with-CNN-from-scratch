#pragma once
#include "Tensor.h"
#include <vector>
#include <unordered_map>

// Stochastic Gradient Descent optimizer with optional momentum and L2 weight decay.
class SGD {
private:
    float learningRate;
    float momentum;
    std::unordered_map<Tensor*, Tensor> velocity; // Per-parameter momentum accumulator, keyed by parameter pointer identity
public:
    SGD(float learningRate, float momentum = 0.0f);

    // params/grads must be parallel vectors (same order, same length) — typically
    // obtained directly from Network::parameters()/gradients() each step.
    void update(std::vector<Tensor*>& params, std::vector<Tensor*>& grads);

    void setLearningRate(float newLearningRate); // Used for manual LR decay/scheduling between epochs
    float getLearningRate() const;
};