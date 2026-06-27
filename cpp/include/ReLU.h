#pragma once
#include "Layer.h"

class ReLU : public Layer
{
private:
    Tensor maskCache; 
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};