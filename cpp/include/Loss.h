#pragma once
#include "Tensor.h"

class Loss
{
public:
    virtual float forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor backward() = 0;
    virtual ~Loss() = default;
};