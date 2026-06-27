#include "SoftMax.h"
#include <cmath>
#include <limits>
#include <algorithm>

Tensor SoftMax::forward(const Tensor& input) {
    size_t N = input.getShape()[0], C = input.getShape()[1]; // Assuming input is 2D (N, C)
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

    outputCache = output; // Cache the output for backward pass
    return output;
}

Tensor SoftMax::backward(const Tensor& gradOutput) {
    size_t N = gradOutput.getShape()[0], C = gradOutput.getShape()[1]; // Assuming gradOutput is 2D (N, C)
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