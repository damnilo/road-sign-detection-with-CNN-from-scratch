#include "SGD.h"

SGD::SGD(float learningRate, float momentum)
    : learningRate(learningRate), momentum(momentum) {}

void SGD::update(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    for(size_t i = 0; i < params.size(); ++i){
        Tensor* param = params[i];
        Tensor* grad = grads[i];

        if(momentum > 0.0f){
            auto it = velocity.find(param);
            if(it == velocity.end()){
                velocity[param] = Tensor::zeros(param->getShape());
                it = velocity.find(param);
            }
            it->second = it->second * momentum + (*grad);
            *param -= it->second * learningRate;
        } else {
            *param -= (*grad) * learningRate;
        }
    }
}