#include "Serialization.h"
#include <fstream>
#include <stdexcept>

void Serialization::saveNetwork(Network& network, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    auto params = network.parameters();
    size_t numLayers = network.numLayers();
    outFile.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

    // Save each layer's parameters
    for (Tensor* param : params) {
        const std::vector<size_t>& shape = param->getShape();
        size_t numDims = shape.size();
        outFile.write(reinterpret_cast<const char*>(&numDims), sizeof(numDims));
        outFile.write(reinterpret_cast<const char*>(shape.data()), numDims * sizeof(size_t));

        const std::vector<float>& data = param->raw();
        size_t dataSize = data.size();
        outFile.write(reinterpret_cast<const char*>(data.data()), dataSize * sizeof(float));
        
    }
}

void Serialization::loadNetwork(Network& network, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    auto params = network.parameters();
    size_t numLayers;
    inFile.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    if(numLayers != network.numLayers()) {
        throw std::runtime_error("Mismatch in number of layers when loading network");
    }

    for(Tensor* param : params) {
        size_t numDims;
        inFile.read(reinterpret_cast<char*>(&numDims), sizeof(numDims));
        std::vector<size_t> shape(numDims);
        inFile.read(reinterpret_cast<char*>(shape.data()), numDims * sizeof(size_t));

        if(shape != param->getShape()) {
            throw std::runtime_error("Mismatch in tensor shape when loading network");
        }

        std::vector<float>& data = param->raw();
        size_t dataSize = data.size();
        inFile.read(reinterpret_cast<char*>(data.data()), dataSize * sizeof(float));
    }

}