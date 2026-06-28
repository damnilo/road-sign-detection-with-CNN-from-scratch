#pragma once
#include "Layer.h"

// Inverted dropout regularization layer. During training, randomly zeroes a
// fraction (`rate`) of activations and scales the survivors by 1/(1-rate) so
// the expected activation magnitude is preserved; this means eval mode needs
// no special-case scaling, just a pass-through.
class Dropout : public Layer
{
private:
    float rate;
    bool training;
    Tensor maskCache; // Per-element scale (0 if dropped, 1/(1-rate) if kept) — reused in backward()

public:
    explicit Dropout(float rate);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    void setTraining(bool isTraining); // Must be set to false before evaluation/inference
    bool isTraining() const;
};