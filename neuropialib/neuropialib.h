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
                return m_network.sizes();
            }
            std::optional<Sizes> load(const Bytes& bytes) {
                const auto map = m_network.load(bytes);
                if(!map) return std::nullopt;
                return m_network.sizes();
            }
            /**
             * @brief Feed values to network, get calculated output.
             * 
             * @param input 
             * @return Values 
             */

            template <typename IT>
            const Values& feed(const IT& begin, const IT& end) const {return m_network.feed(begin, end);}

            /**
             * @brief Access to Neuropia network input layer
             * 
             * @return const Layer& 
             */
            const Layer& network() const {return m_network;}
        private:
            Layer m_network = {};
   };
} // namespace Neuropia
