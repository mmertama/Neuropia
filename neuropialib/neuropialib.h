#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>

/**
 * @brief Interface to use network
 *  
 *  Construct - load - feed
 * 
 */
namespace Neuropia
{
    using Bytes = std::vector<uint8_t>;
    using Values = std::vector<double>;
    struct Sizes {unsigned in_layer; unsigned out_layer;};
    class Network {
        public:
            Network();
            ~Network();
            /**
             * @brief Load a network
             * 
             * @param bytes 
             * @return std::optional<Sizes>, if ok in and output layer sizes. 
             */
            std::optional<Sizes> load(const uint8_t* bytes, size_t sz);
            std::optional<Sizes> load(const Bytes& bytes);
            /**
             * @brief Feed values to network, get calculated output.
             * 
             * @param input 
             * @return Values 
             */
            Values feed(const Values& input) const;
        private:
            class Private;
            std::unique_ptr<Private> m_private;
   };
} // namespace Neuropia
