#include "Network.h"

void Network::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

// Forward pass: each layer's output feeds directly into the next.
Tensor Network::forward(const Tensor& input) {
    Tensor output = input;
    for(auto& layer : layers){
        output = layer->forward(output);
    }
    return output;
}

// Backward pass: walks layers in reverse, since gradients flow from the loss
// back toward the input — mirror image of forward().
Tensor Network::backward(const Tensor& gradOutput) {
    Tensor grad = gradOutput;
    for(auto it = layers.rbegin(); it != layers.rend(); ++it){
        grad = (*it)->backward(grad);
    }
    return grad;
}

// Flattens every layer's parameters() into one list, preserving layer order —
// this order must match gradients()'s order exactly, since the optimizer zips them together.
std::vector<Tensor*> Network::parameters() {
    std::vector<Tensor*> params;
    for(auto& layer : layers){
        auto layerParams = layer->parameters();
        params.insert(params.end(), layerParams.begin(), layerParams.end());
    }
    return params;
}

std::vector<Tensor*> Network::gradients() {
    std::vector<Tensor*> grads;
    for(auto& layer : layers){
        auto layerGrads = layer->gradients();
        grads.insert(grads.end(), layerGrads.begin(), layerGrads.end());
    }
    return grads;
}

size_t Network::numLayers() const {
    return layers.size();
}

Layer* Network::getLayer(size_t index) {
    if(index >= layers.size()){
        throw std::out_of_range("Layer index out of range");
    }
    return layers[index].get();
}