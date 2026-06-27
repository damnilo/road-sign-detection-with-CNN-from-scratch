#include "Network.h"

void Network::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

Tensor Network::forward(const Tensor& input) {
    Tensor output = input;
    for(auto& layer : layers){
        output = layer->forward(output);
    }
    return output;
}

Tensor Network::backward(const Tensor& gradOutput) {
    Tensor grad = gradOutput;
    for(auto it = layers.rbegin(); it != layers.rend(); ++it){
        grad = (*it)->backward(grad);
    }
    return grad;
}

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