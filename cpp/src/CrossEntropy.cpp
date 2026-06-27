#include "CrossEntropy.h"
#include <cmath>

float CrossEntropy::forward(const Tensor& predictions, const Tensor& targets) {
    predictionsCache = predictions; // Cache the predictions for backward pass
    targetsCache = targets;         // Cache the targets for backward pass

    size_t N = predictions.getShape()[0], C = predictions.getShape()[1]; // Assuming predictions is 2D (N, C)
    const float eps = 1e-12f; // Small value to avoid log(0)
    float loss = 0.0f;

    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j < C; ++j){
            float t = targets.at({i, j});
            if(t > 0.0f){ // Only consider the target class
                float p = predictions.at({i, j});
                loss -= t * std::log(std::max(p, eps)); // Use max to avoid log(0)
            }
        }
    }

    return loss / static_cast<float>(N); // Return average loss
}

Tensor CrossEntropy::backward() {
    size_t N = predictionsCache.getShape()[0], C = predictionsCache.getShape()[1]; // Assuming predictions is 2D (N, C)
    Tensor gradInput({N, C});
    const float eps = 1e-12f; // Small value to avoid division by zero

    for(size_t i = 0; i < N; ++i){
        for(size_t j = 0; j < C; ++j){
            float t = targetsCache.at({i, j});
            float p = predictionsCache.at({i, j});
            gradInput.at({i, j}) = (-t / (p + eps)) / (float) N; // Use max to avoid division by zero
        }
    }

    return gradInput * (1.0f / static_cast<float>(N)); // Return average gradient
}