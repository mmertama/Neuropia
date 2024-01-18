#include "neuropialib.h"
#include "../include/neuropia.h"

using namespace Neuropia;

class Network::Private {
public:
    std::optional<Sizes> load(const uint8_t* bytes, size_t sz) {
        const auto& [ok, map] = m_network.load(bytes, sz); // should use std::optional as C++17 should be ok
        if(!ok)
            return std::nullopt;
        return std::make_optional<Sizes>(Sizes{
            static_cast<unsigned>(m_network.size()),
            static_cast<unsigned>(m_network.outLayer()->size())});
    }
    std::optional<Sizes> load(const Bytes& bytes) {
        // map contains creation params, we omit them
        const auto& [ok, map] = m_network.load(bytes); // should use std::optional as C++17 should be ok
        if(!ok)
            return std::nullopt;
        return std::make_optional<Sizes>(Sizes{
            static_cast<unsigned>(m_network.size()),
            static_cast<unsigned>(m_network.outLayer()->size())});
    }
    Values feed(const Values& values) const {return m_network.feed(values.begin(), values.end());}
private:
    Layer m_network;
};

Network::Network() : m_private{std::make_unique<Private>()} {}
Network::~Network() {}
std::optional<Sizes> Network::load(const uint8_t* bytes, size_t sz) {return m_private->load(bytes, sz);}
std::optional<Sizes> Network::load(const Bytes& bytes) {return m_private->load(bytes);}
Values Network::feed(const Values& input) const {return m_private->feed(input);}
