cmake_minimum_required(VERSION 3.15)

project(cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES cpp/bindings.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
pybind11_add_module(cpp
    ${SOURCES}
    cpp/src/Tensor.cpp
    cpp/src/Network.cpp
    cpp/src/Dense.cpp
    cpp/src/Conv2D.cpp
    cpp/src/ReLU.cpp
    cpp/src/SoftMax.cpp
    cpp/src/Flatten.cpp
    cpp/src/MaxPool2D.cpp
    cpp/src/SGD.cpp
    cpp/src/CrossEntropy.cpp
    cpp/src/Serialization.cpp
    cpp/src/Dropout.cpp
)

target_include_directories(cpp PRIVATE cpp/include)
target_link_libraries(cpp PRIVATE OpenMP::OpenMP_CXX)