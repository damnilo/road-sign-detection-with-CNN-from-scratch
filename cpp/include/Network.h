#pragma once
#include "Layer.h"
#include <vector>
#include <memory>

// Container that owns an ordered stack of Layers and drives forward/backward
// propagation through all of them, plus parameter/gradient collection for the optimizer.
class Network
{
private:
    std::vector<std::shared_ptr<Layer>> layers;

public:
    Network() = default;
    Network(const Network&) = delete;            // Layers are heavy and shouldn't be implicitly copied
    Network& operator=(const Network&) = delete;
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;

    void addLayer(std::shared_ptr<Layer> layer);

    Tensor forward(const Tensor& input);          // Runs input through every layer in order
    Tensor backward(const Tensor& gradOutput);     // Runs gradient through every layer in reverse order

    std::vector<Tensor*> parameters();             // Collects parameter pointers from every layer, in order
    std::vector<Tensor*> gradients();              // Collects gradient pointers from every layer, in the same order

    size_t numLayers() const;
    Layer* getLayer(size_t index);                 // Direct access to a specific layer (e.g. to toggle Dropout's training mode)
};