#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>

//Tensor is the fundamental n-dimensional array type used throughout the library.
//Internally backed by a flat std::vector<float> with row-major (C-style) layout;
//shape metadata is tracked seperately and used to compute flat offsets on demand.
class Tensor
{
private:
    std::vector<size_t> shape;
    std::vector<float> data;
    
    //Converts a multi-dimensional index (e.g. {batch, channel, row, col}) into
    //a single offset into the flat 'data' buffer, using row-major strides.
    size_t flattenIndex(const std::vector<size_t>& indices) const;

public:
    Tensor();                                          //Empty tensor (used as a placeholder)
    Tensor(const std::vector<size_t>& shape);          //Allocates and zero-sizes a tensor of the given type
    Tensor(std::initializer_list<size_t> shape);                //Convenience overload for literal shapes

    const std::vector<size_t>& getShape() const;                

    size_t size() const;                //Total number of elements
    size_t dimensions() const;          //Number of axes
    void reshape(const std::vector<size_t>& newShape);  // Changes shape metadata only; total element count must remain the same

    //Multi-dimensional element access, bounds- and dimensions-checked via flattenIndex
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    //Flat (single-index) access into the underlying buffer, bypassing shape entirely.
    //Used internally by layers when shape-aware indexing is not needed
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    void fill(float value);             //Sets every element to a constant
    void randomize(float minValue = -1.0f, float maxValue = 1.0f); //Uniform random init, used for weight initialization
    
    // Direct access to the underlying flat buffer — used for bulk numpy interop
    // and for vectorized operations (e.g. gradient clipping) without going through `at()`.
    std::vector<float>& raw();
    const std::vector<float>& raw() const;

    // Elementwise binary ops. All require identical shapes (no broadcasting except broadcastAdd below).
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Elementwise (Hadamard) product, NOT matrix multiplication
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    Tensor operator*(float scalar) const;
    Tensor operator+(float scalar) const;

    // 2D matrix multiplication: (M,K) x (K,N) -> (M,N). Both operands must be 2D.
    Tensor matmul(const Tensor& other) const;

    // 2D transpose: (M,N) -> (N,M).
    Tensor transpose() const;

    // Reduces a tensor along one axis by summation, removing that axis from the result shape.
    // Used for bias-gradient accumulation (summing gradients across the batch axis).
    Tensor sum(size_t axis) const;

    // Adds a 1D row vector to every row of a 2D tensor (shape (N, C) + shape (C,) -> (N, C)).
    // This is the only broadcasting operation in the library; used to add Dense/Conv biases.
    Tensor broadcastAdd(const Tensor& rowVector) const;

    static Tensor zeros(const std::vector<size_t>& shape); // Allocates a tensor and explicitly fills it with 0
};