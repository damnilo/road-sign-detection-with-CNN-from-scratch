#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Tensor.h"
#include "Layer.h"
#include "Dense.h"
#include "ReLU.h"
#include "Conv2D.h"
#include "Flatten.h"
#include "SoftMax.h"
#include "CrossEntropy.h"
#include "Network.h"
#include "SGD.h"
#include "Serialization.h"
#include "MaxPool2D.h"
#include "Loss.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp, m) {
    m.doc() = "Neural Network Library Bindings";
    m.def("createNetwork", []() { return new Network(); }, py::return_value_policy::take_ownership);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<std::initializer_list<size_t>>())
        .def("getShape", &Tensor::getShape)
        .def("size", &Tensor::size)
        .def("dimensions", &Tensor::dimensions)
        .def("at", (float& (Tensor::*)(const std::vector<size_t>&)) &Tensor::at)
        .def("at", (const float& (Tensor::*)(const std::vector<size_t>&) const) &Tensor::at)
        .def("__getitem__", [](const Tensor& t, const std::vector<size_t>& indices) { return t.at(indices); })
        .def("__setitem__", [](Tensor& t, const std::vector<size_t>& indices, float value) { t.at(indices) = value; })
        .def("raw", [](Tensor& t) { return t.raw(); }, py::return_value_policy::reference)
        .def("reshape", &Tensor::reshape)
        .def("fill", &Tensor::fill)
        .def("randomize", &Tensor::randomize)
        .def_static("zeros", &Tensor::zeros)
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> array) {
            auto buf = array.request();
            std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
            Tensor tensor(shape);
            float* src = static_cast<float*>(buf.ptr);
            std::vector<size_t> idx(shape.size(), 0);
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.at(idx) = src[i];
                for(int j = (int)shape.size() - 1; j >= 0; --j) {
                    if (++idx[j] < shape[j]) break;
                    idx[j] = 0;
                }
            }
            return tensor;
        }))
        .def("to_numpy", [](const Tensor& t) {
            auto shape = t.getShape();
            py::array_t<float> out(shape);
            float* dst = static_cast<float*>(out.request().ptr);
            std::vector<size_t> idx(shape.size(), 0);
            for (size_t i = 0; i < t.size(); ++i) {
                dst[i] = t.at(idx);
                for(int j = (int)shape.size() - 1; j >= 0; --j) {
                    if (++idx[j] < shape[j]) break;
                    idx[j] = 0;
                }
            }
            return out;
        });

    py::classh<Layer>(m, "Layer");

    py::classh<Dense, Layer>(m, "Dense")
        .def(py::init<size_t, size_t>())
        .def("forward", &Dense::forward)
        .def("backward", &Dense::backward)
        .def("parameters", &Dense::parameters)
        .def("gradients", &Dense::gradients);

    py::classh<ReLU, Layer>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward);

    py::classh<Conv2D, Layer>(m, "Conv2D")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>())
        .def("forward", &Conv2D::forward)
        .def("backward", &Conv2D::backward)
        .def("parameters", &Conv2D::parameters)
        .def("gradients", &Conv2D::gradients);

    py::classh<Flatten, Layer>(m, "Flatten")
        .def(py::init<>())
        .def("forward", &Flatten::forward)
        .def("backward", &Flatten::backward);
    
    py::classh<SoftMax, Layer>(m, "SoftMax")
        .def(py::init<>())
        .def("forward", &SoftMax::forward)
        .def("backward", &SoftMax::backward);
    
    py::class_<Loss>(m, "Loss");

    py::class_<CrossEntropy, Loss>(m, "CrossEntropy")
        .def(py::init<>())
        .def("forward", &CrossEntropy::forward)
        .def("backward", &CrossEntropy::backward);
    
    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def("addLayer", &Network::addLayer)
        .def("forward", &Network::forward)
        .def("backward", &Network::backward)
        .def("parameters", &Network::parameters)
        .def("gradients", &Network::gradients)
        .def("numLayers", &Network::numLayers)
        .def("getLayer", &Network::getLayer, py::return_value_policy::reference);

    py::class_<SGD>(m, "SGD")
        .def(py::init<float, float>(), py::arg("learningRate"), py::arg("momentum") = 0.0f)
        .def("update", &SGD::update);
    
    py::class_<Serialization>(m, "Serialization")
        .def_static("saveNetwork", &Serialization::saveNetwork)
        .def_static("loadNetwork", &Serialization::loadNetwork);

    py::classh<MaxPool2D, Layer>(m, "MaxPool2D")
        .def(py::init<size_t, size_t>())
        .def("forward", &MaxPool2D::forward)
        .def("backward", &MaxPool2D::backward);
}