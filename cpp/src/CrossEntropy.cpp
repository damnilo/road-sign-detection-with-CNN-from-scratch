#include "CrossEntropy.h"
#include <cmath>

// Computes -sum(target * log(prediction)) per sample, averaged over the batch.
// Only the target class's term contributes (one-hot targets), and `eps` guards
// against log(0) for predictions that have collapsed to exactly zero.
float CrossEntropy::forward(const Tensor& predictions, const Tensor& targets) {
    predictionsCache = predictions;
    targetsCache = targets;

    size_t N = predictions.getShape()[0], C = predictions.getShape()[1];
    const float eps = 1e-12f;
    float loss = 0.0f;

    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j < C; ++j){
            float t = targets.at({i, j});
            if(t > 0.0f){
                float p = predictions.at({i, j});
                loss -= t * std::log(std::max(p, eps));
            }
        }
    }

    return loss / static_cast<float>(N);
}

// dL/dPrediction = -target / prediction, averaged over the batch.
// Note: this is the gradient with respect to softmax *probabilities*, not logits —
// it's designed to be passed into SoftMax::backward(), which applies the softmax
// Jacobian to convert it into a gradient with respect to logits.
Tensor CrossEntropy::backward() {
    size_t N = predictionsCache.getShape()[0], C = predictionsCache.getShape()[1];
    Tensor gradInput({N, C});
    const float eps = 1e-12f;

    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j < C; ++j){
            float t = targetsCache.at({i, j});
            float p = predictionsCache.at({i, j});
            gradInput.at({i, j}) = (-t / (p + eps));
        }
    }

    return gradInput * (1.0f / static_cast<float>(N));
}