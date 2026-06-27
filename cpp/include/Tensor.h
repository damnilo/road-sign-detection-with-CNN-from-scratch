#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>

class Tensor
{
private:
    std::vector<size_t> shape;
    std::vector<float> data;
    
    size_t flattenIndex(const std::vector<size_t>& indices) const;

public:
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(std::initializer_list<size_t> shape);

    const std::vector<size_t>& getShape() const;

    size_t size() const;
    size_t dimensions() const;
    void reshape(const std::vector<size_t>& newShape);

    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    void fill(float value);
    void randomize(float minValue = -1.0f, float maxValue = 1.0f);
    
    std::vector<float>& raw();
    const std::vector<float>& raw() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    Tensor operator*(float scalar) const;
    Tensor operator+(float scalar) const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor sum(size_t axis) const;
    Tensor broadcastAdd(const Tensor& rowVector) const;

    static Tensor zeros(const std::vector<size_t>& shape);
};