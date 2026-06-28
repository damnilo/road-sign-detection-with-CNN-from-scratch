#pragma once
#include "Tensor.h"

// Abstract base class for loss functions (currently just CrossEntropy).
// Unlike Layer, forward() takes both predictions and targets, and backward()
// takes no arguments — it uses values cached internally during forward().
class Loss
{
public:
    virtual float forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor backward() = 0;
    virtual ~Loss() = default;
};