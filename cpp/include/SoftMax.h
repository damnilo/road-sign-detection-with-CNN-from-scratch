#pragma once
#include "Layer.h"

class SoftMax : public Layer
{
private:
    Tensor outputCache; // Cache the output for backward pass

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override; 
};