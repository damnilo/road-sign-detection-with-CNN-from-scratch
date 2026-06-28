#include "Tensor.h"

#include <random>
#include <numeric>
#include <algorithm>

Tensor::Tensor(){
    // Default-constructed tensor has empty shape and empty data — a valid but
    // "unset" state, used as a placeholder member before a real shape is known.
}

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape) {
    size_t total = 1;
    for(auto s: shape){
        total *= s;
    }
    data.resize(total); // Note: resize() default-initializes floats to 0.0f
}

Tensor::Tensor(std::initializer_list<size_t> shape) : Tensor(std::vector<size_t>(shape)) {
    // Delegates to the vector constructor
}

const std::vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::size() const {
    return data.size();
}

size_t Tensor::dimensions() const {
    return shape.size();
}

// Converts multi-dim indices to a flat offset using row-major strides, computed
// right-to-left so the last axis has stride 1 (standard C-array layout).
size_t Tensor::flattenIndex(const std::vector<size_t>& indices) const {
    if(indices.size() != shape.size()){
        throw std::invalid_argument("Number of indices does not match number of dimensions");
    }

    size_t index = 0;
    size_t multiplier = 1;

    for(int i = shape.size() - 1; i >= 0; --i){
        if(indices[i] >= shape[i]){
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * multiplier;
        multiplier *= shape[i];
    }

    return index;
}

float& Tensor::at(const std::vector<size_t>& indices) {
    return data[flattenIndex(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data[flattenIndex(indices)];
}

float& Tensor::operator[](size_t index) {
    return data[index];
}

const float& Tensor::operator[](size_t index) const {
    return data[index];
}

// Shape-only reshape — total element count must match, since no data is moved or copied.
void Tensor::reshape(const std::vector<size_t>& newShape) {
    size_t newSize = 1;
    for(auto s: newShape){
        newSize *= s;
    }

    if(newSize != data.size()){
        throw std::invalid_argument("New shape must have the same number of elements as the old shape");
    }

    shape = newShape;
}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

// Used for weight initialization. Each layer constructor controls the actual
// [minValue, maxValue] range (e.g. Xavier/He bounds), this just fills uniformly within it.
void Tensor::randomize(float minValue, float maxValue) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(minValue, maxValue);

    for(auto& d: data){
        d = dis(gen);
    }
}

std::vector<float>& Tensor::raw() {
    return data;
}

const std::vector<float>& Tensor::raw() const {
    return data;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for addition");
    }
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for subtraction");
    }
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Elementwise multiply — used for masking (e.g. ReLU/Dropout gradients), not matrix multiplication.
Tensor Tensor::operator*(const Tensor& other) const {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for multiplication");
    }
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for addition");
    }
    for(size_t i = 0; i < data.size(); ++i){
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for subtraction");
    }
    for(size_t i = 0; i < data.size(); ++i){
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] + scalar;
    }
    return result;
}

// Standard 2D matrix multiply: (M,K) x (K,N) -> (M,N), parallelized over the M (row) axis.
// Loop order is i-j-k (row, then K/inner-dim, then N/output-col) so that for each fixed
// (i,j) the scalar `a` is held constant while accumulating across the full output row —
// this is the correct convention; an earlier bug used `j` as both the K-loop and the
// accumulation index, which silently corrupted every Dense/Conv2D forward and backward pass.
Tensor Tensor::matmul(const Tensor& other) const {
    if(shape.size() != 2 || other.shape.size() != 2){
        throw std::invalid_argument("Both tensors must be 2D for matrix multiplication");
    }
    if(shape[1] != other.shape[0]){
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }

    size_t M = shape[0];
    size_t K = shape[1];
    size_t N = other.shape[1];

    Tensor result({M, N});

    const float* A = data.data();
    const float* B = other.data.data();
    float* C = result.data.data();

    #pragma omp parallel for
    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < K; ++j){
            float a = A[i*K + j];
            for(size_t k = 0; k < N; ++k){
                C[i*N + k] += a * B[j*N + k];
            }
        }
    }

    return result;
}

// 2D transpose: (M,N) -> (N,M). Straightforward element swap, parallelized over rows.
Tensor Tensor::transpose() const {
    if(shape.size() != 2){
        throw std::invalid_argument("Tensor must be 2D for transpose");
    }

    Tensor result({shape[1], shape[0]});

    const float* src = data.data();
    float* dst = result.data.data();

    #pragma omp parallel for
    for(size_t i = 0; i < shape[0]; ++i){
        for(size_t j = 0; j < shape[1]; ++j){
            dst[j * shape[0] + i] = src[i * shape[1] + j];
        }
    }

    return result;
}

// Sums the tensor along one axis, collapsing it out of the result shape.
// Used primarily for bias gradients: summing dL/dOutput over the batch axis (axis 0).
Tensor Tensor::sum(size_t axis) const {
    if(axis >= shape.size()){
        throw std::invalid_argument("Axis out of bounds");
    }

    // Fast path: summing over axis 0 of a 2D tensor (the common case for bias gradients)
    // avoids the general N-dimensional index bookkeeping below.
    if(axis == 0){
        Tensor result({shape[1]});
        for(size_t j = 0; j < shape[0]; ++j){
            for(size_t i = 0; i < shape[1]; ++i){
                result[i] += data[j * shape[1] + i];
            }
        }
        return result;
    }

    // General N-dimensional case: build the result shape by dropping `axis`,
    // then walk every element of the source tensor, map its multi-index to the
    // corresponding (axis-removed) index in the result, and accumulate.
    std::vector<size_t> newShape = shape;
    newShape.erase(newShape.begin() + axis);

    Tensor result(newShape);
    std::fill(result.raw().begin(), result.raw().end(), 0.0f);

    std::vector<size_t> indices(shape.size(), 0);
    for(size_t i = 0; i < data.size(); ++i){
        // NOTE: this inner loop computes `flatIndex`/`dimPos` but the result is never
        // used — the actual flat-index calculation happens below via `reduced`/`fi`/`m`.
        // Harmless dead code, safe to remove if cleaning up.
        size_t flatIndex = 0, multiplier = 1;
        for(int j = shape.size() - 1; j >= 0; --j){
            if(j == axis) continue;
            size_t dimPos = (j > axis) ? j - 1 : j;
        }

        // Build the index tuple with `axis` removed, then flatten it against newShape.
        std::vector<size_t> reduced;
        for(size_t j = 0; j < shape.size(); ++j){
            if(j != axis){
                reduced.push_back(indices[j]);
            }
        }
        size_t fi = 0, m = 1;
        for(int j = reduced.size() - 1; j >= 0; --j){
            fi += reduced[j] * m;
            m *= newShape[j];
        }

        result.raw()[fi] += data[i];

        // Odometer-style increment of the multi-index, skipping the reduced axis.
        for(int j = shape.size() - 1; j >= 0; --j){
            if(j == axis) continue;
            if(++indices[j] < shape[j]) break;
            indices[j] = 0;
        }
    }

    return result;
}

// Adds a 1D row vector to every row of a 2D tensor — the only broadcasting op in the
// library. Used to add per-output-channel biases after a Dense or Conv2D matmul.
Tensor Tensor::broadcastAdd(const Tensor& rowVector) const {
    if(rowVector.dimensions() != 1 || rowVector.size() != shape[1]){
        throw std::invalid_argument("Row vector must be 1D and match the number of columns");
    }

    Tensor result(shape);

    const float* a = data.data();
    const float* b = rowVector.data.data();
    float* c = result.data.data();
    for(size_t i = 0; i < shape[0]; ++i){
        for(size_t j = 0; j < shape[1]; ++j){
            c[i * shape[1] + j] = a[i * shape[1] + j] + b[j];
        }
    }

    return result;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    Tensor result(shape);
    result.fill(0.0f);
    return result;
}