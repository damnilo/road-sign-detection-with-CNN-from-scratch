#pragma once
#include "Network.h"
#include <string>

class Serialization {
public:
    static void saveNetwork(Network& network, const std::string& filename);
    static void loadNetwork(Network& network, const std::string& filename);
};