#include "SoftMax.h"
#include <cmath>
#include <limits>
#include <algorithm>

// Numerically-stable softmax: subtract the per-row max before exponentiating
// to avoid overflow, then normalize by the row sum.
Tensor SoftMax::forward(const Tensor& input) {
    size_t N = input.getShape()[0], C = input.getShape()[1];
    Tensor output({N, C});

    for(size_t i = 0; i < N; ++i){
        float maxVal = -std::numeric_limits<float>::infinity();
        for(size_t j = 0; j < C; ++j){
            maxVal = std::max(maxVal, input.at({i, j}));
        }

        float sumExp = 0.0f;
        for(size_t j = 0; j < C; ++j){
            float expVal = std::exp(input.at({i, j}) - maxVal);
            output.at({i, j}) = expVal;
            sumExp += expVal;
        }

        for(size_t j = 0; j < C; ++j){
            output.at({i, j}) /= sumExp;
        }
    }

    outputCache = output;
    return output;
}

// Standard softmax Jacobian-vector product:
//   dL/dInput_j = p_j * (dL/dOutput_j - sum_k(p_k * dL/dOutput_k))
// where p is the cached softmax output from forward(). This correctly accounts
// for every output depending on every input (softmax's outputs aren't independent).
Tensor SoftMax::backward(const Tensor& gradOutput) {
    size_t N = gradOutput.getShape()[0], C = gradOutput.getShape()[1];
    Tensor gradInput({N, C});

    for(size_t i = 0; i < N; ++i){
        float dotProduct = 0.0f;
        for(size_t j = 0; j < C; ++j){
            dotProduct += outputCache.at({i, j}) * gradOutput.at({i, j});
        }

        for(size_t j = 0; j < C; ++j){
            gradInput.at({i, j}) = outputCache.at({i, j}) * (gradOutput.at({i, j}) - dotProduct);
        }
    }

    return gradInput;
}