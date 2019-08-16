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

namespace Neuropia {
    using NeuronType = double;
    class Layer;
    class Neuron;
}

//Not in namespace
std::ostream& operator<< (std::ostream& strm, const std::vector<Neuropia::NeuronType>& values);
std::ostream& operator<<(std::ostream& output, const Neuropia::Layer& layer);
std::ostream& operator<<(std::ostream& output, const Neuropia::Neuron& neuron);


namespace Neuropia {
/**
 * @brief NeuronType
 * Basic type for calculations
 */
/**
 * @brief ValueMap
 * Hash map type for non continous NeuronType data
 */
using ValueMap = std::unordered_map<int, NeuronType>;
/**
 * @brief ValueVector
 * Array type for NeuronType
 */
using ValueVector = std::vector<NeuronType>;

//C++ functions are uncomparable and typedef is not hard, thus we make a wrapper functor to help this
template <typename R, typename ...U>
class NFunction {
public:
    bool operator==(std::nullptr_t) const noexcept {return m_f == nullptr;}
    bool operator!=(std::nullptr_t) const noexcept {return m_f != nullptr;}
    bool operator==(const NFunction& other) const noexcept {return m_name == other.m_name;}
    bool operator!=(const NFunction& other) const noexcept {return m_name != other.m_name;}
    explicit operator bool() const noexcept {return m_f != nullptr;}
    std::function<R (U...)> function() {return m_f;}
    R operator()(U... values) const {return m_f(values...);}
    std::string name() const noexcept {return m_name;}
protected:
    NFunction() {}
    NFunction(std::function<R(U...)> f, const std::string& name):  m_f(f), m_name(name){}
private:
   std::function<R(U...)> m_f = nullptr;
   std::string m_name;
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

class DerivativeFunction : public NFunction<NeuronType, NeuronType> {
public:
   DerivativeFunction();
   DerivativeFunction(void* t) {(void)t;}
   DerivativeFunction(std::function<NeuronType(NeuronType value)> f, const std::string& name) :  NFunction(f, name){}
};

/**
  Default Activationfunctions
  */


#define ACTIVATION_FUNCTION(name, f) const ActivationFunction name(f, #name);
#define DERIVATIVE_FUNCTION(name, f) const DerivativeFunction name(f, #name);

constexpr double LeakyReLuFactor = 0.05; //too small makes backpropagation not working
constexpr double EluFactor = 1.0;

ACTIVATION_FUNCTION(signumFunction, [](NeuronType value) -> NeuronType{
    return value < 0.0 ? -1.0 : value > 0.0 ? 1.0 : 0.0;
});

ACTIVATION_FUNCTION(binaryFunction, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return value >= 1.0 ? 0.0 : 1.0;
});

ACTIVATION_FUNCTION(sigmoidFunction, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return 1.0 / (1.0 + std::exp(-value));
});

ACTIVATION_FUNCTION(reLuFunction, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return std::max(value, LeakyReLuFactor * value);
});

ACTIVATION_FUNCTION(eluFunction, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return value < 0.0 ? EluFactor * (std::exp(value) - 1.0) : value;
});

#ifndef DEFAULT_AF
#define DEFAULT_AF sigmoidFunction
#endif

/**
  Default Derivative function
  */
DERIVATIVE_FUNCTION(sigmoidFunctionDerivative, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return value * (1.0 - value);
});

DERIVATIVE_FUNCTION(reLuFunctionDerivative, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return value > 0.0 ? 1.0 : LeakyReLuFactor;
});

DERIVATIVE_FUNCTION(eluFunctionDerivative, [](Neuropia::NeuronType value) -> Neuropia::NeuronType {
    return value > 0.0 ? 1.0 : EluFactor * std::exp(value);
});

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
 * @param af
 * @return
 */
DerivativeFunction derivativeMap(ActivationFunction af);


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
     * @param mf
     * ActivationFunction for ths neuron
     */
    Neuron(ActivationFunction af,
           const ValueVector& weights = ValueVector(),
           NeuronType bias = 1) noexcept : m_af(af), m_weights(weights), m_bias(bias) {
    }


    /**
     * @brief setActivationFunction
     * Set ActivationFunction
     * @param af
     * @return
     */
    void setActivationFunction(ActivationFunction af) {
        m_af = af.function();
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

    NeuronType feed(const ValueVector& inputs) const;

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
    void save(std::ofstream& stream) const;

    /**
     * @brief load
     * @param stream
     * @return
     */
    void load(std::ifstream& stream);

    /**
     * @brief operator <<
     * @param output
     * @param layer
     * @return
     */
    friend std::ostream& ::operator<<(std::ostream& output, const Neuron& neuron);

private:
    std::function<NeuronType (NeuronType)> m_af = nullptr;
    ValueVector m_weights;
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
     * @param proto
     */
    Layer(size_t count, const ActivationFunction& activationFunction, const Neuron& proto) noexcept;

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
     * @param proto
     */
    void fill(size_t count, const Neuron& proto);

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
     * @param proto
     * @return
     */
    Layer& join(size_t count, const Neuron& proto);

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
     * @param proto
     * @return
     */
    Layer& join(const std::initializer_list<int>& topology, const Neuron& proto);

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
     * @param proto
     * @return
     */
    Layer& join(IteratorIt begin, IteratorIt end, const Neuron& proto) {
        for(auto it = begin; it != end; it++) {
            join(*it, proto);
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

    template<typename IteratorIt>
    /**
     * @brief feed
     * @param values
     * @return
     */
    const ValueVector& feed(IteratorIt begin, IteratorIt end) const {
        neuropia_assert(m_activationFunction);
        if(!isInput()) {
            const ValueVector values(begin, end);
            neuropia_assert(m_outBuffer.size() >= m_neurons.size());
            for(size_t i = 0; i < m_neurons.size(); i++) {
                const auto& n = m_neurons[i];
                neuropia_assert(n.isActive());
                m_outBuffer[i] = n.feed(values);
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
     * @param df
     * @return
     */
    bool train(IteratorItInput inputs, IteratorItOutput expectedOutputs, double learningRate, double lambdaL2, const DerivativeFunction& dfp = nullptr) {
        const auto seed =
#ifndef RANDOM_SEED
                static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
        RANDOM_SEED
#endif
        ;
        std::default_random_engine gen(seed);
        dropout(gen);

        const auto out = feedTrain(inputs, inputs + m_neurons.size()); //go forward first
        ValueVector expectedValues(out.size());
        std::copy(expectedOutputs, expectedOutputs + out.size(), expectedValues.begin());
        const auto df = dfp == nullptr ? Neuropia::derivativeMap(m_activationFunction) : dfp;
        return backpropagation(out, expectedValues, learningRate, lambdaL2, df);
    }

    /**
     * @brief dropout
     * @param dropoutRate
     * @param inherit
     */
    void dropout(double dropoutRate, bool inherit = true);

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
     * @brief isInpu
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
    void save(std::ofstream& stream, const std::unordered_map<std::string, std::string>& meta = {}) const;


    /**
     * @brief load
     * @param stream
     * @return
     */
    std::tuple<bool, std::unordered_map<std::string, std::string>> load(std::ifstream& stream);

    /**
     * @brief merge
     * @param other
     * @param factor
     */
    void merge(const Layer& other, double factor);

    /**
     * @brief compare
     * @param other
     * @return
     */
    int compare(const Layer& other) const;

    /**
     * @brief The InitStategy enum
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
     * @param af
     */
    void setActivationFunction(const ActivationFunction& af) {m_activationFunction = af;}

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

 protected:
    void loadLayer(std::ifstream& stream);

    Layer* previousLayer(Layer* current);
    const Layer* previousLayer(const Layer* current) const;
    bool backpropagation(const ValueVector& out, const ValueVector& expected, double learningRate, double lambdaL2, const DerivativeFunction& df);
    void dropout(std::default_random_engine& gen);

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
                    const auto out = n.feed(values);
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
    std::vector<Neuron> m_neurons;
    std::unique_ptr<Layer> m_next;
    Layer* m_prev = nullptr;
    ActivationFunction m_activationFunction = nullptr;
    double m_dropOut = 0.0;
    mutable ValueVector m_outBuffer;
};


/**
 * @brief initStrategyMap
 * @param af
 * @return
 */
Layer::InitStrategy initStrategyMap(ActivationFunction af);

}



#endif // NEUROPIA

