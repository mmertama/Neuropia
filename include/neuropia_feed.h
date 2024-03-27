#pragma once

#include "neuropia.h"
#include <string_view>
#include <array>

namespace Neuropia {
    template<const uint8_t* D, size_t SZ, typename SType = float>
    
    /// @brief Read only feed
    /// Compile time Feed forward network
    class Feed {
       
    private:

        constexpr static inline uint8_t H5[] = {'N', 'E', 'U', '0', '0', '0', '0', '5'};
        constexpr static inline auto layer_count_pos = sizeof(H5) + 2;
        constexpr static inline auto meta_count_pos = sizeof(H5) + 3;
        
        using Str = std::pair<size_t, size_t>; // Str is position to string begin and end
        using Meta = std::pair<Str, Str>; // Meta is key and value Str
    
        using Weights = std::pair<size_t, size_t>; //Range of weights
        using Neuron = std::tuple<Weights, size_t>; //Range of neuron + bias position
        using LayerInfo = std::tuple<size_t, Weights, Str>;

        enum {LAYER_SIZE, LAYER_NEURONS, LAYER_ACTIVATION};
        enum {NEURON_WEIGHTS, NEURON_BIAS};

    public:
        /// @brief There should not be need to call this
        constexpr Feed() {
            static_assert(SZ > sizeof(H5));
            static_assert(is_equal<sizeof(H5)>(D, H5));
            constexpr auto save_type = static_cast<SaveType>(get_8(sizeof(H5)));
            static_assert(save_type == SaveType::Float);
            constexpr auto is_big_endian = static_cast<bool>(get_8(sizeof(H5) + 1));
            static_assert(isBigEndian() == is_big_endian); // TODO this should indicate that read_32 can be a straight read
            static_assert(layer_count() >= 3); // at least one hidden layer
        }

        /// @brief Number of defined parameters used in network creation
        /// @return 
        static constexpr auto parameter_count() {
            return static_cast<size_t>(get_8(meta_count_pos));
        }

        /// @brief Number of layers 
        /// @return 
        static constexpr size_t layer_count() {
            return static_cast<size_t>(get_8(layer_count_pos));
        }
    
        /// @brief get parameters
        /// @param index 
        /// @return key and value pair
        static constexpr auto parameter(size_t index) {
            const auto offset_to = meta_offset(index);
            const auto meta = make_meta(offset_to);
            const auto key = to_string(meta.first);
            const auto value = to_string(meta.second);
            return std::pair<std::string_view, std::string_view>{key, value};
        }

        /// @brief in layer size 
        /// @return constexpr size_t 
        static constexpr size_t in_layer_size() {
            return std::get<LAYER_SIZE>(get_layer_info(0));
        }

        /// @brief out layer size
        /// @return 
        static constexpr size_t out_layer_size() {
            return std::get<LAYER_SIZE>(get_layer_info(layer_count() - 1));
        }   
    
    private:
        
        static constexpr auto str_len(const Str& str) {
            return str.second - str.first;
        }

        static constexpr auto to_string(const Str& str) {
            return std::string_view(reinterpret_cast<const char*>(D + str.first), str_len(str));
        }

        template<size_t SUB_SZ>    
        static constexpr bool is_equal(const uint8_t* a, const uint8_t* b) {
            for(auto i = 0; i < SUB_SZ ; ++i)
                if(a[i] != b[i])
                    return false;
            return true;         
        }

        static constexpr uint8_t get_8(size_t pos) {
            return D[pos];        
        }

                // little endian!
        static constexpr uint32_t get_32(size_t pos) {
            return static_cast<uint32_t>(D[pos + 0])    |
            static_cast<uint32_t>(D[pos + 1]) << 8      |
            static_cast<uint32_t>(D[pos + 2]) << 16     |
            static_cast<uint32_t>(D[pos + 3]) << 24;
        }

        // not constexpr due runtime cast
        static inline SType read_real(size_t pos) {
            return *(reinterpret_cast<const SType*>(&D[pos]));
        }

        
        /// @brief Return pointers to begin and end of data
        static constexpr Str make_str(size_t pos) {
            return Str{ pos + 1, pos + 1 + get_8(pos) };
        }


        /// @brief Return key value pairs of Str
        static constexpr Meta make_meta(size_t pos) {
            return Meta{ make_str(pos), make_str(make_str(pos).second) };
        }

        static constexpr auto name_to_function(const Str& function_name) {
            const auto name = to_string(function_name);
            if (signumFunction.name() == name)
                return signumFunction;
            else if (binaryFunction.name() == name)
                return binaryFunction;
            else if (sigmoidFunction.name() == name)
                return sigmoidFunction;
            else if ( reLuFunction.name() == name)
                return reLuFunction;
            else if (eluFunction.name() == name)
                return eluFunction;
            else
                return ActivationFunction{};
        }

        // layer neurons and end position
        static constexpr LayerInfo make_layer_info(size_t pos) {
            const auto af_name = make_str(pos);
            const auto drop_pos = af_name.second;
            pos = drop_pos + sizeof(uint32_t); // 1st dropout
            const auto sz = get_32(pos);
            pos += sizeof(uint32_t);  // size
            auto n_pos = pos;
            for(auto i = 0U; i < sz; ++i) {
                const auto neuron_info = make_neuron_info(n_pos);
                n_pos = std::get<NEURON_WEIGHTS>(neuron_info).second;
            }
            // I read this value here so I don't have to do this for each feed, and memory is (about) the same (or twice more or less depending on configuration given) :-)
            // drop is not used for feed const SType drop_out = read(drop_pos);
            return {sz, {pos, n_pos}, af_name};
        }
    
        static constexpr Neuron make_neuron_info(size_t pos) {
            const auto bias_pos = pos;
            pos += sizeof(SType);
            const auto sz = get_32(pos);
            pos += sizeof(uint32_t);
            const auto end_pos = pos + sizeof(SType) * sz;
            return {{pos, end_pos}, bias_pos};  
        } 


        static constexpr size_t meta_offset(size_t index) {
            constexpr auto meta_count = parameter_count();
            auto pos = meta_count_pos + 1;
            for(size_t i = 0; i < index; ++i) {
                pos = make_meta(pos).second.second;
            }
            return pos;
        }


        static constexpr LayerInfo make_layer_info_at(size_t at, size_t offset) {
            const LayerInfo info = make_layer_info(offset);
            return at == 0 ?  info : make_layer_info_at(at - 1, std::get<LAYER_NEURONS>(info).second);
        }

        static constexpr LayerInfo get_layer_info(size_t index) {
            constexpr auto m_off = meta_offset(parameter_count()); // after metadata
            const LayerInfo info = make_layer_info_at(index, m_off);
            return info;
        }




        template<typename IT>
        static SType feed_neuron(const Neuron& neuron, IT begin, IT end, const ActivationFunction& activation_function) {
            auto sum = read_real(std::get<NEURON_BIAS>(neuron));
            const auto sz = std::distance(begin, end);
            auto pos = std::get<NEURON_WEIGHTS>(neuron).first;
            for(auto i = 0; i < sz; ++i) { // difference type is signed - hence int is ok, auto my know better
                sum += (read_real(pos) * *(begin + i));
                pos += sizeof(SType);
            }
            return activation_function(sum);
        }    


    public:
        using OutValues = std::array<SType, out_layer_size()>;            
        template<typename IT>
        /**
         * @brief Feed neural network
         * 
         * @param begin 
         * @param end 
         * @return OutValues 
         */
        static OutValues feed(IT begin, IT end) {
            static_assert(std::is_same<typename IT::value_type, SType>::value);
            std::array<SType, std::get<LAYER_SIZE>(get_layer_info(1))> a_buffer; // - a buffer is 1st write buffer... the maximum buffer size as next layer < previous, and input is outside
            const auto af_1 = name_to_function(std::get<LAYER_ACTIVATION>(get_layer_info(1)));
            constexpr auto off_1 = std::get<LAYER_NEURONS>(get_layer_info(1)); // 0 is input layer , structured bind cannot be constexpr
            auto it = a_buffer.begin();
            auto pos = off_1.first;
            while(pos < off_1.second) {
                const auto neuron = make_neuron_info(pos);
                pos = std::get<NEURON_WEIGHTS>(neuron).second;
                const auto result = feed_neuron(neuron, begin, end, af_1);
                *it = result;
                ++it;
            }
            
            std::array<SType, std::get<LAYER_SIZE>(get_layer_info(2))> b_buffer; // the maximum buffer size as next layer < previous, and input is outside
            
            auto write_begin = b_buffer.begin();
            auto read_begin = a_buffer.begin();
            auto read_end = it; // this is where the last write ended!

            for(auto i = 2U; i < layer_count(); ++i) { 
                const auto& [sz, off, af_name] = get_layer_info(i);
                const auto activation_function = name_to_function(af_name);
                it = write_begin;
                pos = off.first;
                while(pos < off.second) {
                    const auto neuron = make_neuron_info(pos);
                    pos = std::get<NEURON_WEIGHTS>(neuron).second;
                    const auto result = feed_neuron(neuron, read_begin, read_end, activation_function);
                    *it = result;
                    ++it;
                }
                std::swap(write_begin, read_begin);
                read_end = it; // this is where the last write ended!
            }
            OutValues out; // a bit unnecessary copy, but I hope compiler would deal this
            neuropia_assert(out.size() == static_cast<size_t>(std::distance(read_begin, it)));  
            std::copy(read_begin, it, out.begin());
            return out;
        }

    /**
     * @brief get a layer info 
     * 
     * @param index 
     * @return tuple of layer size and activation function name 
     */
    static constexpr auto layer_info(size_t index) {
            const auto& [sz, off, af_str] = get_layer_info(index);
            return std::make_tuple(sz, to_string(af_str));
        }         

    using NeuronData = std::tuple<std::vector<SType>, SType>;
    /// @brief get a single neuron - not much use beyond debug and testing (slow, allocate)
    /// @param layer_index 
    /// @param neuron_index 
    /// @return 
    static std::optional<NeuronData> neuron_info(size_t layer_index, size_t neuron_index) {
            const auto& [sz, off, af_str] = get_layer_info(layer_index);
            size_t index = 0;
            auto pos = off.first;
            while(pos < off.second) {
                const auto& [weights, bias_pos] = make_neuron_info(pos);
                if(index == neuron_index) {
                    // copy values to dynamic buffer
                    std::vector<SType> values;
                    values.reserve((weights.second - weights.first) / sizeof(SType));
                    for(auto w = weights.first; w < weights.second; w += sizeof(SType)) {
                        values.push_back(read_real(w));
                    }
                    const auto bias = read_real(bias_pos);
                    return NeuronData{std::move(values), bias};
                }
                ++index;
                pos = weights.second;
            }
            return std::nullopt;    
        }
    };
}
