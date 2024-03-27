#ifndef NEUROPIA
#define NEUROPIA

#include <functional>
#include <unordered_map>
#include <algorithm>
#include <initializer_list>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <optional>
#include <fstream>

/**
 * Namespace Neuropia
 */

// C assert.h is debug, I wanted to have a release also substitution
inline bool doAssert(const std::string& s, int line, const char* file, const std::string& extra = "") {
    if(extra.length() == 0) {
        std::cerr << "assert:" << s << " at:" << file << " line:" << line << std::endl;
    } else {
        std::cerr << "assert:" << s << "(" << extra  << ")" << " at:" << file <<  " line:"  << line << std::endl;
    }
    std::abort();
}

#ifndef NEUROPIA_DEBUG
#if (!defined(NDEBUG) || defined(_DEBUG)) && !defined(__OPTIMIZE__)
    #define NEUROPIA_DEBUG
#endif
#endif

#ifdef NEUROPIA_DEBUG
#define neuropia_assert(x) ((x) || doAssert(#x, __LINE__, __FILE__))
#define neuropia_assert_x(x, extra) ((x) || doAssert(#x, __LINE__, __FILE__, extra))
#define neuropia_assert_always(x, extra) ((x) || doAssert(#x, __LINE__, __FILE__, extra))
#else
#define neuropia_assert(x)
#define neuropia_assert_x(x, extra)
#define neuropia_assert_always(x, extra) ((x) || doAssert(#x, __LINE__, __FILE__, extra))
#endif

#ifndef NEUROPIA_TYPE
#define NEUROPIA_TYPE double
#endif

namespace Neuropia {
    /// @brief @Value type used in Neuropia (NEUROPIA_TYPE is compile time defined- see a CMakeLists.txt)
    using NeuronType = NEUROPIA_TYPE;
    class Layer;
    class Neuron;
    class StreamBase;
}

//Not in namespace
std::ostream& operator<< (std::ostream& stream, const std::vector<Neuropia::NeuronType>& values);
std::ostream& operator<<(std::ostream& output, const Neuropia::Layer& layer);
std::ostream& operator<<(std::ostream& output, const Neuropia::Neuron& neuron);


namespace Neuropia {
/**
 * @brief NeuronType
 * Basic type for calculations
 */
/**
 * @brief ValueMap
 * Hash map type for non continuous NeuronType data
 */
using ValueMap = std::unordered_map<int, NeuronType>;
/**
 * @brief ValueVector
 * Array type for NeuronType
 */
using ValueVector = std::vector<NeuronType>;

/// @brief Network creation parameters map type - See Params
using MetaInfo = std::unordered_map<std::string, std::string>;

/**
 * @brief Save data types. NeuronType is a Neuropia::NeuronType, others are C++ floating point data types 
 * 
 * 
 */
enum class SaveType : uint8_t {
    SameAsNeuronType, Double, Float, LongDouble
};

/**
 * @brief Stored header information
 * 
 */
struct Header {
    /// @brief Value type used
    const SaveType saveType;
    /// @brief Number of layers
    const unsigned layers;
    /// @brief Endianness used
    const bool bigEndian;
};


inline constexpr
bool isBigEndian() {
   // before supports C++20, then this can be written reliably
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return false;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return true;
#else
    union {
        uint32_t i;
        uint8_t c[4];
    } bint = {0x01020304};
    return bint.c[0] == 1; // luck
    #endif
}

/**
 * @brief Verify that file is Neuropia file
 * 
 * @param filename 
 * @return std::optional<SaveType, unsigned> where unsigned is number of layers 
 */
std::optional<Header> isValidFile(const std::string& filename);

/// @brief In and out layer dimensions
struct Sizes {unsigned in_layer; unsigned out_layer;};

template <typename R, typename ...U>
/// @brief helper class
/// C++ functions are incomparable and typedef is not hard, thus we make a wrapper functor to help this
class NFunction {
public:
/// @cond
    bool operator==(std::nullptr_t) const noexcept {return m_f == nullptr;}
    bool operator!=(std::nullptr_t) const noexcept {return m_f != nullptr;}
    bool operator==(const NFunction& other) const noexcept {return m_name == other.m_name;}
    bool operator!=(const NFunction& other) const noexcept {return m_name != other.m_name;}
    explicit operator bool() const noexcept {return m_f != nullptr;}
    std::function<R (U...)> function() {return m_f;}
    R operator()(U... values) const {return m_f(values...);}
    constexpr std::string_view name() const noexcept {return m_name;}
protected:
    NFunction() {}
    NFunction(std::function<R(U...)> f, const std::string& name):  m_f(f), m_name(name){}
 /// @endcond
private:
   std::function<R(U...)> m_f = nullptr;
   std::string m_name = {};
};

/**
 * @brief ActivationFunction
 * Activation function type
 */
class ActivationFunction : public NFunction<NeuronType, NeuronType> {
public:
   ActivationFunction() {}
   ActivationFunction(void* t) {(void)t;}
   ActivationFunction(std::function<NeuronType(NeuronType value)> f, const std::string& name) :  NFunction(f, name){}
};

/**
 * @brief DerivativeFunction
 * 
 */
class DerivativeFunction : public NFunction<NeuronType, NeuronType> {
public:
   DerivativeFunction();
   DerivativeFunction(void* t) {(void)t;}
   DerivativeFunction(std::function<NeuronType(NeuronType value)> f, const std::string& name) :  NFunction(f, name){}
};


#define ACTIVATION_FUNCTION(name, f) const ActivationFunction name(f, #name);
#define DERIVATIVE_FUNCTION(name, f) const DerivativeFunction name(f, #name);

constexpr NeuronType LeakyReLuFactor = static_cast<NeuronType>(0.05); //too small makes backpropagation not working
constexpr NeuronType EluFactor = static_cast<NeuronType>(1.0);

/// @brief Signum function
ACTIVATION_FUNCTION(signumFunction, [](NeuronType value) noexcept -> NeuronType {
    return value < 0.0 ? -1.0 : value > 0.0 ? 1.0 : 0.0;
})

/// @brief Binary function
ACTIVATION_FUNCTION(binaryFunction, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return value >= 1.0 ? 0.0 : 1.0;
})

/// @brief Sigmoid function
ACTIVATION_FUNCTION(sigmoidFunction, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return static_cast<NeuronType>(1.0 / (1.0 + std::exp(-value)));
})

/// @brief ReLu function
ACTIVATION_FUNCTION(reLuFunction, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return std::max(value, LeakyReLuFactor * value);
})

/// @brief eLu function
ACTIVATION_FUNCTION(eluFunction, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return static_cast<NeuronType>(value < 0.0 ? EluFactor * (std::exp(value) - 1.0) : value);
})



/**
  @brief Default Derivative function
*/

#ifndef DEFAULT_AF
#define DEFAULT_AF sigmoidFunction
#endif

/// @brief Sigmoid derivative function 
DERIVATIVE_FUNCTION(sigmoidFunctionDerivative, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return value * (static_cast<NeuronType>(1.0) - value);
})

/// @brief ReLu derivative function 
DERIVATIVE_FUNCTION(reLuFunctionDerivative, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return static_cast<NeuronType>(value > 0.0 ? 1.0 : LeakyReLuFactor);
})

/// @brief eLu derivative function 
DERIVATIVE_FUNCTION(eluFunctionDerivative, [](Neuropia::NeuronType value) noexcept -> Neuropia::NeuronType {
    return static_cast<NeuronType>(value > 0.0 ? 1.0 : EluFactor * std::exp(value));
})

/**
 * @brief normalize
 * Helper function to normalize input [-1, 1]
 * @param value
 * @param min
 * @param max
 * @return
 */

inline
NeuronType normalize(NeuronType value, NeuronType min, NeuronType max) {
    neuropia_assert(min < max);
    return (value - min) / (max - min);
}

/**
 * @brief derivativeMap
 * @param activation_function
 * @return
 */
DerivativeFunction derivativeMap(ActivationFunction activation_function);


/**
 * @brief The Neuron class
 * Represents a single neuron
 */
class Neuron {

public:

    Neuron() = default;

    /**
     * @brief Neuron
     * @param weights
     * Weights used for this neuron
     * @param bias
     * Bias used for this neuron
     * @param activation_function
     * ActivationFunction for this neuron
     */
    Neuron(ActivationFunction activation_function,
           const ValueVector& weights = ValueVector(),
           NeuronType bias = 1) noexcept : m_af(activation_function), m_weights(weights), m_bias(bias) {
    }


    /**
     * @brief setActivationFunction
     * Set ActivationFunction
     * @param activation_function
     * @return
     */
    void setActivationFunction(ActivationFunction activation_function) {
        m_af = activation_function.function();
    }

    /**
     * @brief isActive
     * @return
     */
    bool isActive() const {
        return m_af != nullptr;
    }
    /**
     * @brief setBiases
     * @param biases
     * @return
     */
    void setBias(NeuronType value) {
        m_bias = value;
    }

    /**
     * @brief setWeights
     * @param weights
     * @return
     */
    void setWeights(const ValueVector& weights) {
        neuropia_assert(isActive());
        m_weights = weights;
    }

    /**
     * @brief setWeights
     * @param weights
     * @return
     */
    void setWeights(ValueVector&& weights) {
        m_weights = std::move(weights);
    }

    /**
     * @brief setWeight
     * @param index
     * @param value
     * @return
     */
    void setWeight(size_t index, NeuronType value) {
        neuropia_assert(isActive());
        m_weights[index] = value;
    }

    /**
     * @brief hasWeights
     * @return
     */
    bool hasWeights() const {return !m_weights.empty();}

    /**
     * @brief feed
     * Set input from array
     * @param inputs
     * @return
     */

    template<typename IT>
    NeuronType feed(IT begin, IT end) const;

    /**
     * @brief size
     * @return
     */
    size_t size() const {return m_weights.size();}

    /**
     * @brief bias
     * @return
     */
    NeuronType bias() const {return m_bias;}

    /**
     * @brief weight
     * @param index
     * @return
     */
    NeuronType weight(size_t index) const {
        neuropia_assert(isActive());
        return m_weights[index];
    }

    NeuronType weight_d(size_t index) const {
        return m_weights[index];
    }

    /**
     * @brief save
     * @param stream
     */
    void save(std::ofstream& stream, SaveType saveType = SaveType::SameAsNeuronType) const;

    /**
     * @brief load
     * @param stream
     * @param saveType 
     * @return
     */
    bool load(std::ifstream& stream, SaveType saveType = SaveType::SameAsNeuronType);

    /**
     * @brief 
     * @param bytes 
     * @param saveType 
     * @return 
     */
    bool load(const std::vector<uint8_t>& bytes, SaveType saveType = SaveType::SameAsNeuronType);

    /**
     * @brief operator <<
     * @param output
     * @param layer
     * @return
     */

    /**
     * @brief memory consumption
     * 
     * @return size_t 
     */
    size_t consumption() const;

    friend std::ostream& ::operator<<(std::ostream& output, const Neuron& neuron);

    // @internal
    [[nodiscard]] bool loadNeuron(StreamBase& stream, SaveType saveType);
private:
    
    std::function<NeuronType (NeuronType)> m_af = nullptr;
    ValueVector m_weights = {};
    NeuronType m_bias = 1;
};

/**
 * @brief The Layer class
 */
class Layer {
public:

    /**
     * @brief Layer
     * @param activationFunction
     */
    Layer(const ActivationFunction& activationFunction = DEFAULT_AF) : m_activationFunction(activationFunction) {}

    /**
     * @brief Layer
     * @param list
     * @param activationFunction
     */
    Layer(const std::initializer_list<Neuron>& list, const ActivationFunction& activationFunction = DEFAULT_AF) noexcept;

    /**
     * @brief Layer
     * @param count
     * @param activationFunction
     * @param prototype
     */
    Layer(size_t count, const ActivationFunction& activationFunction, const Neuron& prototype) noexcept;

    /**
     * @brief Layer
     * @param count
     * @param activationFunction
     */
    Layer(size_t count, const ActivationFunction& activationFunction = DEFAULT_AF) noexcept : Layer(count, activationFunction, Neuron(activationFunction)) {}

    /**
     * @brief Layer
     * @param other
     */
    Layer(Layer&& other) noexcept;

    /**
     * @brief Layer
     * @param other
     */
    Layer(const Layer& other) noexcept;

    /**
     * @brief operator =
     * @param other
     * @return
     */
    Layer& operator=(Layer&& other) noexcept;

    /**
     * @brief operator =
     * @param other
     * @return
     */
    Layer& operator=(const Layer& other) noexcept;

    /**
     * @brief destructor
     */
    virtual ~Layer();
    /**
     * @brief append
     * @param neuron
     * @return
     */
    void append(const Neuron& neuron);

    /**
     * @brief fill
     * @param count
     * @param prototype
     */
    void fill(size_t count, const Neuron& prototype);

    /**
     * @brief join
     * Adds a new layer
     * @param next
     * @return
     */
    Layer& join(Layer* next);

    /**
     * @brief join
     * @param count
     * @param prototype
     * @return
     */
    Layer& join(size_t count, const Neuron& prototype);

    /**
     * @brief join
     * @param count
     * @return
     */
    Layer& join(size_t count) {
        return join(count, Neuron(m_activationFunction));
    }

    /**
     * @brief join
     * @param topology
     * @param prototype
     * @return
     */
    Layer& join(const std::initializer_list<int>& topology, const Neuron& prototype);

    /**
     * @brief join
     * @param topology
     * @return
     */
    Layer& join(const std::initializer_list<int>& topology) {
        return join(topology, Neuron(m_activationFunction));
    }


    template<typename IteratorIt>
    /**
     * @brief join
     * @param begin
     * @param end
     * @param prototype
     * @return
     */
    Layer& join(IteratorIt begin, IteratorIt end, const Neuron& prototype) {
        for(auto it = begin; it != end; it++) {
            join(static_cast<size_t>(*it), prototype);
        }
        return *this;
    }

    template<typename IteratorIt>
    /**
     * @brief join
     * @param begin
     * @param end
     * @return
     */
    Layer& join(IteratorIt begin, IteratorIt end) {
        return join(begin, end, Neuron(m_activationFunction));
    }

    template<typename IT>
    /**
     * @brief feed
     * @param values
     * @return
     */
    const ValueVector& feed(IT begin, IT end) const;

    /**
     * @brief feed
     * @param vec
     * @return
     */
    const ValueVector& feed(const ValueVector& vec) {
        return feed(vec.begin(), vec.end());
    }

    /**
     * @brief randomize
     * @param min
     * @param max
     */
    void randomize(NeuronType min = -1, NeuronType max = 1);


    template<typename IteratorItInput, typename IteratorItOutput>
    /**
     * @brief train
     * @param inputs
     * @param expectedOutputs
     * @param learningRate
     * @param derivativeFunction
     * @return
     */
    bool train(IteratorItInput inputs, IteratorItOutput expectedOutputs, NeuronType learningRate, NeuronType lambdaL2, const DerivativeFunction& derivativeFunction = nullptr) {
        const auto seed =
#ifndef RANDOM_SEED
                static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
        RANDOM_SEED
#endif
        ;
        std::default_random_engine gen(seed);
        dropout(gen);

        const auto out = feedTrain(inputs, inputs + static_cast<int>(m_neurons.size())); //go forward first
        ValueVector expectedValues(out.size());
        std::copy(expectedOutputs, expectedOutputs + static_cast<int>(out.size()), expectedValues.begin());
        const auto derivativeFunction_ptr = derivativeFunction == nullptr ? Neuropia::derivativeMap(m_activationFunction) : derivativeFunction;
        return backpropagation(out, expectedValues, learningRate, lambdaL2, derivativeFunction_ptr);
    }

    /**
     * @brief dropout
     * @param dropoutRate
     * @param inherit
     */
    void dropout(NeuronType dropoutRate, bool inherit = true);

    /**
     * @brief inverseDropout
     * @param inherit
     */
    void inverseDropout(bool inherit = true);

    /**
     * @brief size
     * @return
     */
    size_t size() const {return m_neurons.size();}

    /**
     * @brief isInput
     * @return
     */
    bool isInput() const {return m_prev == nullptr;}

    /**
     * @brief isOutput
     * @return
     */
    bool isOutput() const {return m_next == nullptr;}

    /**
     * @brief save
     * @param stream
     */
    void save(std::ofstream& stream, const MetaInfo& meta = {}, SaveType saveType = SaveType::SameAsNeuronType) const;


    /**
     * @brief load
     * @param stream
     * @return
     */
    std::optional<MetaInfo> load(std::ifstream& stream);

    /**
     * @brief load
     * @param stream
     * @return
     */
    std::optional<MetaInfo> load(const std::vector<uint8_t>& stream);

    /**
     * @brief load
     * @param stream
     * @return
     */
    std::optional<MetaInfo> load(const uint8_t* bytes, size_t sz);

    /**
     * @brief merge
     * @param other
     * @param factor
     */
    void merge(const Layer& other, NeuronType factor);

    /**
     * @brief compare
     * @param other
     * @return
     */
    int compare(const Layer& other) const;

    /**
     * @brief The InitStrategy enum
     */
    enum class InitStrategy{Norm, Logistic, ReLu};

    /**
     * @brief init
     * @param strategy
     */
    void initialize(InitStrategy strategy);

    /**
     * @brief get
     * @param offset
     * @return
     */
    Layer* get(int offset);

    /**
     * @brief get
     * @param offset
     * @return
     */
    const Layer* get(int offset) const;

    /**
     * @brief operator <<
     * @param output
     * @param layer
     * @return
     */
    friend std::ostream& ::operator<<(std::ostream& output, const Layer& layer);

    /**
     * @brief begin
     * @return
     */
    std::vector<Neuron>::const_iterator begin() const {return m_neurons.begin();}

    /**
     * @brief end
     * @return
     */
    std::vector<Neuron>::const_iterator end() const {return m_neurons.end();}

    /**
     * @brief setActivationFunction
     * @param activation_function
     */
    void setActivationFunction(const ActivationFunction& activation_function) {m_activationFunction = activation_function;}

    /**
     * @brief activationFunction
     * @return
     */
    ActivationFunction activationFunction() const {return m_activationFunction;}

    /**
     * @brief operator []
     * @param index
     * @return
     */
    const Neuron& operator[](size_t index) const {return *(begin() + static_cast<int>(index));}

    /**
     * @brief isValid
     * @param testNext
     * @return
     */
    bool isValid(bool testNext = true) const;

    /**
     * @brief outLayer
     * @return
     */
    Layer* outLayer();
    const Layer* outLayer() const;

    /**
     * @brief in and out sizes
     * 
     * @return Sizes 
     */
    Sizes sizes() const;

    /**
     * @brief memory consumption
     * 
     * @return size_t 
     */
    size_t consumption(bool cumulative) const;

    /**
     * @brief get input layer (most cases assert(layer == layer->inputLayer()))
     * 
     * @return const Layer* 
     */
    const Layer* inputLayer() const;


    /**
     * @brief next
     * @return
     */
    const Layer* next() const {return get(1);}
    
 protected:
    [[nodiscard]] bool loadLayer(StreamBase& stream, SaveType saveType, unsigned layer_count);

    Layer* previousLayer(Layer* current);
    const Layer* previousLayer(const Layer* current) const;

    bool backpropagation(const ValueVector& out, const ValueVector& expected, NeuronType learningRate, NeuronType lambdaL2, const DerivativeFunction& derivativeFunction);
    void dropout(std::default_random_engine& gen);

    std::optional<MetaInfo> doLoad(StreamBase& stream);

    template<typename IteratorIt>
    const ValueVector& feedTrain(IteratorIt begin, IteratorIt end) const {
        if(!isInput()) {
            const auto p = 1.0  - m_dropOut;
            for(size_t i = 0; i < m_neurons.size(); i++) {
                const auto& n = m_neurons[i];
                if(n.isActive()) {
                    const auto sz = static_cast<size_t>(std::distance(begin, end));
                    ValueVector values(sz);
                    const auto prevBegin = previousLayer(this)->begin();
                    for(auto j = 0U; j < sz; j++) {
                        values[j] = (prevBegin + j)->isActive() ? *(begin + j) : 0;
                    }
                    const auto out = n.feed(values.begin(), values.end());
                    m_outBuffer[i] = out * p;
                } else {
                    m_outBuffer[i] = 0;
                }
            }
        } else {
            for(auto it = begin; it != end; it++) {
                const auto index = static_cast<unsigned>(std::distance(begin, it));
                m_outBuffer[index] = m_neurons[index].isActive() ? *it : 0;  //copy value only if corresponding neuron is active
            }
        }
        if(m_next != nullptr) {
            return m_next->feedTrain(m_outBuffer.begin(), m_outBuffer.end());
        }
        return  m_outBuffer;

    }

private:
    std::vector<Neuron> m_neurons = {};
    std::unique_ptr<Layer> m_next = nullptr;
    Layer* m_prev = nullptr;
    ActivationFunction m_activationFunction = nullptr;
    NeuronType m_dropOut = 0.0;
    mutable ValueVector m_outBuffer = {};
};


/**
 * @brief initStrategyMap
 * @param activation_function
 * @return
 */
Layer::InitStrategy initStrategyMap(ActivationFunction activation_function);

template<typename IT>
NeuronType Neuron::feed(IT begin, IT end) const {
    neuropia_assert(m_af);
    NeuronType sum = m_bias;
    const auto sz = static_cast<size_t>(std::distance(begin, end));
    neuropia_assert(m_weights.size() >= sz);
    for(size_t i = 0; i < sz; i++) {
        sum += (m_weights[i] * *(begin + static_cast<typename IT::difference_type>(i)));
    }
    return m_af(sum);
}

    template<typename IT>
    const ValueVector& Layer::feed(IT begin, IT end) const {
        neuropia_assert(m_activationFunction);
        if(!isInput()) {
            neuropia_assert(m_outBuffer.size() >= m_neurons.size());
            for(size_t i = 0; i < m_neurons.size(); i++) {
                const auto& n = m_neurons[i];
                neuropia_assert(n.isActive());
                m_outBuffer[i] = n.feed(begin, end);
            }
        } else {
            neuropia_assert(static_cast<size_t>(std::distance(begin, end)) <= m_outBuffer.size());
            std::copy(begin, end, m_outBuffer.begin());
        }
        if(m_next != nullptr) {
            return m_next->feed(m_outBuffer.begin(), m_outBuffer.end());
        }
        return  m_outBuffer;
    }

}



#endif // NEUROPIA

