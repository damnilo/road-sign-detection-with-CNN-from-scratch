#pragma omp parallel for
#include "Tensor.h"

#include <random>
#include <numeric>
#include <algorithm>

Tensor::Tensor(){

}

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape) {
    size_t total = 1;

    for(auto s: shape){
        total *= s;
    }

    data.resize(total);
}

Tensor::Tensor(std::initializer_list<size_t> shape) : Tensor(std::vector<size_t>(shape)) {

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

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            float a = A[i*K + j];
            for(size_t k = 0; k < K; ++k){
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    return result;
}

Tensor Tensor::transpose() const {
    if(shape.size() != 2){
        throw std::invalid_argument("Tensor must be 2D for transpose");
    }

    Tensor result({shape[1], shape[0]});

    const float* src = data.data();
    float* dst = result.data.data();

    for(size_t i = 0; i < shape[0]; ++i){
        for(size_t j = 0; j < shape[1]; ++j){
            dst[j * shape[0] + i] = src[i * shape[1] + j];
        }
    }

    return result;
}

Tensor Tensor::sum(size_t axis) const {
    if(axis >= shape.size()){
        throw std::invalid_argument("Axis out of bounds");
    }

    if(axis == 0){
        Tensor result({shape[1]});

        for(size_t j = 0; j < shape[0]; ++j){
            for(size_t i = 0; i < shape[1]; ++i){
                result[i] += data[j * shape[1] + i];
            }
        }

        return result;
    }

    std::vector<size_t> newShape = shape;
    newShape.erase(newShape.begin() + axis);

    Tensor result(newShape);
    std::fill(result.raw().begin(), result.raw().end(), 0.0f);

    std::vector<size_t> indices(shape.size(), 0);
    for(size_t i = 0; i < data.size(); ++i){
        size_t flatIndex = flattenIndex(indices);
        result.raw()[flatIndex] += data[i];

        // Increment indices
        for(int j = shape.size() - 1; j >= 0; --j){
            if(j == axis) continue;
            indices[j]++;
            if(indices[j] < shape[j]){
                break;
            } else {
                indices[j] = 0;
            }
        }
    }

    return result;
}

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