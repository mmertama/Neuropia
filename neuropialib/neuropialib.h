#pragma once

#include <vector>
#include <string>
#include <optional>
#include "neuropia.h"

/**
 * @brief Interface to use network
 *  
 *  Construct - load - feed
 * 
 */
namespace Neuropia
{
    using Bytes = std::vector<uint8_t>;
    using Values = std::vector<NeuronType>;
    struct Sizes {unsigned in_layer; unsigned out_layer;};
    class Network {
        public:
            Network() {}
            ~Network() {}
            /**
             * @brief Load a network
             * 
             * @param bytes 
             * @return std::optional<Sizes>, if ok in and output layer sizes. 
             */
            std::optional<Sizes> load(const uint8_t* bytes, size_t sz) {
                const auto map = m_network.load(bytes, sz);
                if(!map) return std::nullopt;
                return sizes();
            }
            std::optional<Sizes> load(const Bytes& bytes) {
                const auto map = m_network.load(bytes);
                if(!map) return std::nullopt;
                return sizes(); 
            }
            /**
             * @brief Feed values to network, get calculated output.
             * 
             * @param input 
             * @return Values 
             */

            template <typename IT>
            const Values& feed(const IT& begin, const IT& end) const {{return m_network.feed(begin, end);}}
        private:
            Sizes sizes() const {
                return Sizes{
                    static_cast<unsigned>(m_network.size()),
                    static_cast<unsigned>(m_network.outLayer()->size())};
            }
        private:
            Layer m_network;
   };
} // namespace Neuropia
