#pragma once
#include "Tensor.h"
#include "Loss.h"

// Categorical cross-entropy loss for one-hot encoded targets, expected to be
// paired with a preceding SoftMax layer (predictions should already be probabilities).
class CrossEntropy : public Loss
{
private:
    Tensor predictionsCache;
    Tensor targetsCache;

public:
    float forward(const Tensor& predictions, const Tensor& targets);
    Tensor backward();
};