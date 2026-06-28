#include "Dropout.h"
#include <random>

namespace {
    // thread_local so this would stay safe if training were ever parallelized across layers.
    thread_local std::mt19937 rng(std::random_device{}());
}

Dropout::Dropout(float rate) : rate(rate), training(true) {}

// In eval mode (or rate == 0), this is a no-op pass-through — inverted dropout's
// train-time scaling means no compensating scale-down is needed at inference.
Tensor Dropout::forward(const Tensor& input) {
    if (!training || rate <= 0.0f) {
        return input;
    }

    maskCache = Tensor(input.getShape());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float scale = 1.0f / (1.0f - rate);

    for (size_t i = 0; i < input.size(); ++i) {
        bool keep = dist(rng) > rate;
        maskCache[i] = keep ? scale : 0.0f;
    }

    Tensor output(input.getShape());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * maskCache[i];
    }

    return output;
}

// Reuses the exact same mask generated in forward() — gradients flow only through
// units that survived the forward pass, scaled identically.
Tensor Dropout::backward(const Tensor& gradOutput) {
    if (!training || rate <= 0.0f) {
        return gradOutput;
    }

    Tensor gradInput(gradOutput.getShape());
    for (size_t i = 0; i < gradOutput.size(); ++i) {
        gradInput[i] = gradOutput[i] * maskCache[i];
    }

    return gradInput;
}

void Dropout::setTraining(bool isTraining) {
    training = isTraining;
}

bool Dropout::isTraining() const {
    return training;
}