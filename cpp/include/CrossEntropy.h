#pragma once
#include "Tensor.h"
#include "Loss.h"

class CrossEntropy : public Loss
{
private:
    Tensor predictionsCache; // Cache the predictions for backward pass
    Tensor targetsCache;     // Cache the targets for backward pass

public:
    float forward(const Tensor& predictions, const Tensor& targets);
    Tensor backward();
};