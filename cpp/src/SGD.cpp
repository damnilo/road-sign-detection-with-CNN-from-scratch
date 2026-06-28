#include "SGD.h"

SGD::SGD(float learningRate, float momentum)
    : learningRate(learningRate), momentum(momentum) {}

// For each parameter:
//   effectiveGrad = grad + weightDecay * param          (L2 regularization, decoupled-equivalent for plain SGD)
//   velocity      = momentum * velocity + effectiveGrad  (only if momentum > 0)
//   param        -= learningRate * velocity              (or learningRate * effectiveGrad if no momentum)
//
// `velocity` persists across calls, keyed by the parameter's pointer — this relies
// on `params`/`grads` pointing at the same underlying Tensor objects every step
// (true as long as the caller fetches them once per training run, not per batch).
void SGD::update(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    const float weightDecay = 1e-4f;
    for(size_t i = 0; i < params.size(); ++i){
        Tensor* param = params[i];
        Tensor effectiveGrad = (*grads[i]) + (*param) * weightDecay;

        if(momentum > 0.0f){
            auto it = velocity.find(param);
            if(it == velocity.end()){
                velocity[param] = Tensor::zeros(param->getShape());
                it = velocity.find(param);
            }
            it->second = it->second * momentum + effectiveGrad;
            *param -= it->second * learningRate;
        } else {
            *param -= effectiveGrad * learningRate;
        }
    }
}

void SGD::setLearningRate(float newLearningRate) {
    learningRate = newLearningRate;
}

float SGD::getLearningRate() const {
    return learningRate;
}