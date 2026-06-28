#pragma once
#include "Network.h"
#include <string>

// Saves/loads a Network's parameters (weights + biases of every layer) to/from
// a raw binary file. Architecture (layer types, sizes) is NOT stored — the
// caller must reconstruct an identical Network via create_network() before loading.
class Serialization {
public:
    static void saveNetwork(Network& network, const std::string& filename);
    static void loadNetwork(Network& network, const std::string& filename);
};