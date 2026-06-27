#pragma once
#include "Layer.h"
#include <vector>
#include <memory>

class Network
{
private:
    std::vector<std::shared_ptr<Layer>> layers;

public:
    Network() = default;
    Network(const Network&) = delete; // Disable copy constructor
    Network& operator=(const Network&) = delete; // Disable copy assignment
    Network(Network&&) = default; // Enable move constructor
    Network& operator=(Network&&) = default; // Enable move assignment

    void addLayer(std::shared_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput);
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    size_t numLayers() const;
    Layer* getLayer(size_t index);
};