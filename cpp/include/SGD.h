#pragma once
#include "Tensor.h"
#include <vector>
#include <unordered_map>

class SGD {
private:
    float learningRate;
    float momentum;
    std::unordered_map<Tensor*, Tensor> velocity; // For momentum   
public:
    SGD(float learningRate, float momentum = 0.0f);

    void update(std::vector<Tensor*>& params, std::vector<Tensor*>& grads);
};