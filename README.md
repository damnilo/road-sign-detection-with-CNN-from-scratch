# Road Sign Classification with a CNN from Scratch (C++ & Python)

## Overview

This project implements a **Convolutional Neural Network (CNN) from scratch** for traffic sign classification using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

Dataset can be found on this link:

 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 

Unlike frameworks such as PyTorch or TensorFlow, all neural network components—including convolution, pooling, activation functions, batch normalization, dropout, optimization, loss computation, and backpropagation—are implemented manually in **C++**. The C++ library is exposed to Python through **pybind11**, while Python is used for dataset handling, training, evaluation, and visualization.

---

## Features

* CNN implementation from scratch in C++
* Forward and backward propagation
* Convolutional, pooling, and fully connected layers
* ReLU activation
* Batch Normalization
* Dropout regularization
* Softmax + Cross Entropy loss
* SGD optimizer
* Model serialization (save/load trained networks)
* Python bindings using pybind11
* Training and validation pipeline
* Test dataset evaluation
* Group-aware train/validation split to prevent data leakage

---

## Project Structure

```
.
├── cpp/
│   ├── include/          # Header files
│   ├── src/              # C++ implementation
│   └── bindings.cpp      # Python bindings
│
├── model.py              # Network architecture and training
├── load_dataset.py       # Dataset loading utilities
├── test.py               # Test set evaluation
├── CMakeLists.txt
└── README.md
```

---

## Network Architecture

The implemented CNN consists of:

```
Input (3×32×32)
        │
Conv2D
        │
BatchNorm
        │
ReLU
        │
MaxPool
        │
Conv2D
        │
BatchNorm
        │
ReLU
        │
MaxPool
        │
Flatten
        │
Linear
        │
ReLU
        │
Dropout
        │
Linear
        │
Softmax
```

---

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** consisting of **43 traffic sign classes**.

Images are:

* resized to **32×32**
* converted to RGB
* normalized
* stored in channel-first format `(N, C, H, W)`

Training and validation sets are generated using a **group-aware stratified split**, ensuring images from the same recording sequence are never split between both sets.

---

## Building

### Requirements

* C++17
* CMake
* Python 3
* pybind11
* NumPy
* OpenCV
* pandas

Build the C++ extension:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Training

Run:

```bash
python model.py
```

The best-performing model is automatically saved as:

```
best_model.bin
```

---

## Testing

Evaluate the trained model on the official test dataset:

```bash
python test.py
```

Example output:

```
Test Accuracy: 94.77%
```

---

## Implemented Components

### Layers

* Conv2D
* Linear
* MaxPool2D
* Flatten
* BatchNorm
* Dropout
* ReLU

### Loss

* Cross Entropy

### Optimizer

* Stochastic Gradient Descent (SGD)

### Utilities

* Tensor class
* Serialization
* Dataset loading
* Evaluation

---

## Technologies

* C++
* Python
* pybind11
* OpenCV
* NumPy
* pandas
* CMake

---

## Future Improvements

* Adam optimizer
* Learning rate scheduling
* Data augmentation
* GPU acceleration
* Additional activation functions
* Residual architectures

---

## License

This project was developed for educational purposes as part of a university course in Artificial Intelligence.
