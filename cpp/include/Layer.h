#ifndef LAYER_H
#define LAYER_H
#include "Tensor.h"
#include <vector>

class Layer
{
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;
    virtual std::vector<Tensor*> parameters() { return {}; };
    virtual std::vector<Tensor*> gradients() { return {}; };
    virtual ~Layer() = default;
};

#endif // LAYER_H