#include "neuropia.h"
#include <random>
#include <iostream>
#include <fstream>
#include "matrix.h"

using namespace Neuropia;


std::ostream& operator << (std::ostream& strm, const std::vector<Neuropia::NeuronType>& values) {
    strm << '[';
    if(values.size() > 0) {
        for(size_t i = 0; i < values.size() - 1; i++) {
            strm << values[i] << ", ";
        }
        strm << values[values.size() - 1];
    }
    strm << ']';
    return strm;
}

std::ostream& operator<<(std::ostream& output, const Neuron& neuron) {
    output << '<' << std::endl;
    output << static_cast<bool>(neuron.m_af) << std::endl;
    output << neuron.m_weights;
    output << neuron.m_bias << std::endl;
    output << '>' << std::endl;
    return output;
}

std::ostream& operator<<(std::ostream& output, const Layer& layer) {
    output << '{' << std::endl;
    output << layer.m_activationFunction.name() << std::endl;
    for(const auto& n : layer.m_neurons)
        output << n;
    output << '}' << std::endl;
    if(layer.m_next)
        output << *layer.m_next;
    return output;
}


DerivativeFunction Neuropia::derivativeMap(ActivationFunction af) {
    if(sigmoidFunction == af) return sigmoidFunctionDerivative;
    if(reLuFunction == af) return reLuFunctionDerivative;
    if(eluFunction == af) return eluFunctionDerivative;
    return nullptr;
}

Layer::InitStrategy Neuropia::initStrategyMap(ActivationFunction af) {
    if(sigmoidFunction == af) return Layer::InitStrategy::Logistic;
    if(reLuFunction == af) return Layer::InitStrategy::ReLu;
    if(eluFunction == af) return Layer::InitStrategy::ReLu;
    return Layer::InitStrategy::Norm;
}


NeuronType Neuron::feed(const ValueVector& inputs) const {
    neuropia_assert(m_af);
    NeuronType sum = m_bias;
    for(size_t i = 0; i < inputs.size(); i++) {
        sum += (m_weights[i] * inputs[i]);
    }
    return m_af(sum);
}

void Neuron::save(std::ofstream& stream) const {
    const auto sz = m_weights.size();
    stream.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
    for(const auto& w : m_weights) {
        stream.write(reinterpret_cast<const char*>(&w), sizeof(NeuronType));
    }
    stream.write(reinterpret_cast<const char*>(&m_bias), sizeof(NeuronType));
}

void Neuron::load(std::ifstream& stream) {
    m_weights.clear();
    size_t sz;
    stream.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
    m_weights.resize(sz);
    for(size_t s = 0; s < sz; s++) {
        NeuronType w;
        stream.read(reinterpret_cast<char*>(&w), sizeof(NeuronType));
        setWeight(s, w);
    }
    NeuronType b;
    stream.read(reinterpret_cast<char*>(&b), sizeof(NeuronType));
    setBias(b);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////7


Layer::Layer(const std::initializer_list<Neuron>& list, ActivationFunction activationFunction) noexcept : m_neurons(list), m_activationFunction(activationFunction), m_outBuffer(list.size()) {
}

Layer::Layer(size_t count, ActivationFunction activationFunction, const Neuron& proto) noexcept : m_activationFunction(activationFunction){
    m_neurons.resize(static_cast<unsigned>(count));
    std::fill(m_neurons.begin(), m_neurons.end(), proto);
    m_outBuffer.resize(m_neurons.size());
}

Layer::Layer(Layer&& other) noexcept :
    m_neurons(std::move(other.m_neurons)),
    m_next(std::move(other.m_next)),
    m_activationFunction(other.m_activationFunction),
    m_outBuffer(m_neurons.size()){
    if(m_next) {
        m_next->m_prev = this;
    }
}

Layer::Layer(const Layer& other) noexcept:
    m_neurons(other.m_neurons),
    m_next(other.m_next != nullptr ? new Layer(*other.m_next) : nullptr),
    m_activationFunction(other.m_activationFunction),
    m_outBuffer(m_neurons.size()) {
    if(m_next) {
        m_next->m_prev = this;
    }
}

Layer& Layer::join(size_t count, const Neuron& proto) {
    return join(new Layer(count, m_activationFunction, proto));
}

Layer& Layer::join(Layer* next) {
    if(m_next) {
        return m_next->join(next);
    }
    m_next.reset(next);
    const ValueVector w(m_neurons.size());
    for(auto& n : next->m_neurons) {
        if(!n.hasWeights()) {
            n.setWeights(w);
        }
    }
    next->m_prev = this;
    return *next;
}

Layer& Layer::join(const std::initializer_list<int>& topology, const Neuron& proto) {
    return join(topology.begin(), topology.end(), proto);
}

void Layer::append(const Neuron& neuron) {
    m_neurons.push_back(neuron);
    m_outBuffer.resize(m_neurons.size());
}

void Layer::fill(size_t count, const Neuron& proto) {
    m_neurons.resize(count);
    std::fill(m_neurons.begin(), m_neurons.end(), proto);
    m_outBuffer.resize(m_neurons.size());
}

void Layer::randomize(NeuronType min, NeuronType max) {
    if(!isInput()) {  //actually not needed as setweights wont do notting for input layers
        const auto seed =
#ifndef RANDOM_SEED
                static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
        RANDOM_SEED
#endif
        ;
        std::default_random_engine gen(seed);
        std::uniform_real_distribution<> dis(min, max);

        for(auto& n : m_neurons) {
            for(size_t i = 0; i < n.size(); i++) {
                n.setWeight(i, dis(gen));
            }
        }
        for(auto& n : m_neurons) { //for certain testability reasons we have second loop to set biases
            n.setBias(dis(gen));
        }
    } else {
        for(auto& n : m_neurons) {
            n.setBias(0);
        }
    }
    if(m_next) {
        m_next->randomize(min, max);
    }
}


Layer* Layer::outLayer() {
    if(m_next == nullptr) {
        return this;
    }
    return m_next->outLayer();
}

Layer* Layer::previousLayer(Layer* current) {
    return current->m_prev;
}

const Layer* Layer::previousLayer(const Layer* current) const {
    return current->m_prev;
}


 //This train class implements backpropagation function
 //see://www.youtube.com/watch?v=QJoa0JYaX1I - there are several episode
 //video how that works, therefore only the most basic comments are injected here that may
 //help you the implementation vs. explanation on video
bool Layer::backpropagation(const ValueVector& outValues, const ValueVector& expectedValues, double learningRate, double lambdaL2, DerivativeFunction df) {

//expected values as Matrix
    const auto expected = Matrix<NeuronType>::fromArray(expectedValues, Matrix<NeuronType>::VecDir::row);

//get results as matrices
    auto lastValues = Matrix<NeuronType>::fromArray(outValues, Matrix<NeuronType>::VecDir::row);
//get error if actual output is too big, error is negative
    auto errors = expected - lastValues;

//we go backwards
    auto lastLayer = outLayer();

    if(lastLayer == this) {
        return false;    //sanity
    }

//get layer as Matrix
    auto layerData = Matrix<NeuronType>::fromArray(previousLayer(lastLayer)->m_outBuffer, Matrix<NeuronType>::VecDir::row);

    for(;;) {

        // y is already a sigmoid value  - the function is derivated  sigmoidfunction
        // if s(x) =  1 / (1 + e^-x) then s`(x) = s(x)(1 - s(x)), but since given y is already s(x)
        // the derivated value can be written as
        const auto gradientDelta = lastValues.map(df);
        //not that * operands here are elemental multipcations, not matrix muls
        auto gradients = learningRate * errors * gradientDelta;

#ifdef NEUROPIA_DEBUG
        if(gradients.reduce<bool>(false, [](bool a, auto r){return a || std::isnan(r) || std::isinf(r);}))
            return false;
#endif

        if(!gradients.isValid())
            return false;

        //set lastlayer bias, if neuron would be matrix this would be just B += G
        neuropia_assert(gradients.cols() == 1 && lastLayer->m_neurons.size() == gradients.rows());
        for(auto i = 0U; i < gradients.rows(); i++) {
            auto& n = lastLayer->m_neurons[i];
            if(n.isActive()) { //only if the weight is connected from an active neuron
                const auto g = gradients(0, i);
                n.setBias(n.bias() + g);
            }
        }


        const auto layerDataTransposed = layerData.transpose();

        if(lambdaL2 > 0.0) {
            const auto L2 = gradients.reduce<double>(0, [](auto a, auto r){return a + (r * r);}) /
                static_cast<double>(gradients.rows());
            const auto l = lambdaL2 * L2;
            gradients.mapThis([l](auto v){return v - l;});
        }

        const auto layerDeltas = Matrix<NeuronType>::multiply(gradients, layerDataTransposed);

        const auto prevLayer = previousLayer(lastLayer);
        auto weightsData =
            Matrix<NeuronType>::fromArray(lastLayer->m_neurons,
                                          prevLayer->size(), //fully connected, amount of weights is prev layer neurons
        [prevLayer](const Neuron & n, Matrix<NeuronType>::index_type index) {
            return n.isActive() && (*prevLayer)[index].isActive() ? n.weight(index) : 0.0;
        });



        const auto weightsDataTransposed = weightsData.transpose();
        const auto weights = weightsData + layerDeltas;

        // just copy matrix back to network
        neuropia_assert(weights.rows() == lastLayer->size() && lastLayer->size()  > 0 && lastLayer->m_neurons[0].size() == weights.cols());

        for(auto j = 0U; j < weights.rows(); j++) {
            auto& n = lastLayer->m_neurons[j];
            if(n.isActive()) {
                for(auto i = 0U; i < weights.cols() ; i++) {
                    if((*prevLayer)[i].isActive())
                        n.setWeight(i, weights(i, j));
                }
            }
        }

        //new errors for new gradient
        errors = Matrix<NeuronType>::multiply(weightsDataTransposed, errors);

        //next layer to go
        lastLayer = previousLayer(lastLayer);

        if(lastLayer == nullptr || lastLayer->isInput()) {
            break; //we hit the input layer
        }

        lastValues = layerData; //layerdata is output values from previous (or actually next :-) layer
        //get previous layer weights
        layerData = Matrix<NeuronType>::fromArray(previousLayer(lastLayer)->m_outBuffer, Matrix<NeuronType>::VecDir::row);

    }
    return true;
}


constexpr char H[] = {'N', 'E', 'U', '0', '0', '0', '0', '1'};
void Layer::save(std::ofstream& strm) const {
    if(isInput()) {
        strm.write(H, sizeof(H));
        const auto nl = static_cast<int>(m_activationFunction.name().length());
        strm.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
        strm.write(m_activationFunction.name().data(), nl);
        const auto sz = m_neurons.size();
        strm.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    } else {
        const auto sz = m_neurons.size();
        strm.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        for(const auto& n : m_neurons) {
            n.save(strm);
        }
    }
    if(m_next) {
        m_next->save(strm);
    }
}

bool Hcomp(const char* h) {
    for(auto i = 0U; i < sizeof(H); i++)
        if(h[i] != H[i]) {
            return false;
        }
    return true;
}

Layer::Layer(std::ifstream& strm, ActivationFunction activationFunction, const Neuron& proto, bool isIn) noexcept : m_activationFunction(activationFunction) {
    if(isIn) {
        char hdr[sizeof(H)];
        strm.read(hdr, sizeof(H));
        neuropia_assert_always(Hcomp(hdr), "Corrupted import stream or wrong version");

        size_t namel = 0;
        strm.read(reinterpret_cast<char*>(&namel), sizeof(int));
        std::string name(namel, '0');
        strm.read(&name[0], static_cast<int>(namel));


        if(signumFunction.name() == name)
            m_activationFunction = signumFunction;
        else if(binaryFunction.name() == name)
            m_activationFunction = binaryFunction;
        else if(sigmoidFunction.name() == name)
            m_activationFunction = sigmoidFunction;
        else if(reLuFunction.name() == name)
            m_activationFunction = reLuFunction;
        else if(eluFunction.name() == name)
            m_activationFunction = eluFunction;

       size_t count = 0;
       strm.read(reinterpret_cast<char*>(&count), sizeof(count));
       fill(count, proto);
    } else {
        size_t count = 0;
        strm.read(reinterpret_cast<char*>(&count), sizeof(count));
        fill(count, proto);
        for(auto& n : m_neurons) {
            n.load(strm);
            n.setActivationFunction(m_activationFunction);
        }
    }
    if(!strm.eof()) {
        auto layer = new Layer(strm, m_activationFunction, proto, false);
        if(layer->size() > 0) {
            m_next.reset(layer);
            m_next->m_prev = this;
        } else {
            delete layer; //bit clumsy
        }
    }
}

Layer& Layer::operator=(Layer&& other) noexcept {
    m_neurons = std::move(other.m_neurons);
    m_outBuffer.resize(m_neurons.size());
    m_next = std::move(other.m_next);
    m_activationFunction = std::move(other.m_activationFunction);
    if(m_next) {
        m_next->m_prev = this;
    }
    return *this;
}

Layer& Layer::operator=(const Layer& other) noexcept {
    m_neurons = other.m_neurons;
    m_outBuffer.resize(m_neurons.size());
    m_activationFunction = other.m_activationFunction;
    if(other.m_next) {
        m_next.reset(new Layer(*other.m_next));
    }
    if(m_next) {
        m_next->m_prev = this;
    }
    return *this;
}

void Layer::merge(const Layer& other, double factor) {
    neuropia_assert(other.size() == size());
    neuropia_assert(factor >= 0 && factor <= 1.0);
    if(!isInput()) {
        for(auto n = 0U ; n < m_neurons.size(); n++) {
            auto& nThis = m_neurons[n];
            auto& nOther = other.m_neurons[n];
            for(auto i = 0U; i < nThis.size(); i++) {
                nThis.setWeight(i, nThis.weight(i) * (1. - factor)
                                + nOther.weight(i) * factor);
            }
            nThis.setBias(nThis.bias() * (1 - factor) + nOther.bias() * factor);
        }
    }
    if(m_next) {
        neuropia_assert(other.m_next);
        m_next->merge(*other.m_next, factor);
    }
}


int Layer::compare(const Layer& other) const {
    if(other.size() < size()) {
        return std::numeric_limits<int>::min();
    }
    if(other.size() > size()) {
        return std::numeric_limits<int>::max();
    }
    for(auto n = 0U ; n < m_neurons.size(); n++) {
        const auto& nThis = m_neurons[n];
        const auto& nOther = other.m_neurons[n];
        for(auto i = 0U; i < nThis.size(); i++) {
            if(nOther.weight(i) < nThis.weight(i)) {
                return -1;
            }
            if(nOther.weight(i) > nThis.weight(i)) {
                return 1;
            }
        }
    }
    if(m_next) {
        if(!other.m_next) {
            return 1;
        }
        return m_next->compare(*other.m_next);
    }
    return other.m_next ? -2 : 0;
}

Layer::~Layer() {
}

void Layer::initialize(InitStrategy strategy) {
    if(!isInput()) {  //actually not needed as setweights wont do notting for input layers
        double r = 0;
        switch (strategy) {
        case Layer::InitStrategy::Norm:
            r = 1.0;
            break;
        case Layer::InitStrategy::Logistic:
             r = std::sqrt(6.0 / (m_neurons.size() + m_prev->m_neurons.size()));
            break;
        case Layer::InitStrategy::ReLu:
             r = std::sqrt(2.0) *  std::sqrt(6.0 / (m_neurons.size() + m_prev->m_neurons.size()));
            break;
        }

        unsigned seed =
#ifndef RANDOM_SEED
                static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
                RANDOM_SEED
#endif
                ;
        std::default_random_engine rd(seed);  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-r, r);

        for(auto& n : m_neurons) {
            for(size_t i = 0; i < n.size(); i++) {
                n.setWeight(i, dis(gen));
            }
        }
        for(auto& n : m_neurons) { //for certain testability reasons we have second loop to set biases
            n.setBias(dis(gen));
        }
    } else {
        for(auto& n : m_neurons) {
          //  for(size_t i = 0; i < n.size(); i++) {
          //      n.setWeight(i, 1.0);
          //  }
            n.setBias(0);
        }
    }
    if(m_next) {
        m_next->initialize(strategy);
    }
}

void Layer::dropout(std::default_random_engine& gen) {
    if(m_dropOut > 0.0) {
        if(!isOutput()) { // outputs are not dropped
            const unsigned sz = static_cast<unsigned>(m_neurons.size());
            const auto dropCount = static_cast<unsigned>(static_cast<double>(sz * m_dropOut));
            std::vector<bool> set(sz);
            std::fill(set.begin(), set.end(), false);
            auto count = 0U;
            while(count < dropCount) {
                auto index = gen() % sz;
                while(set[index]) {
                    ++index;
                if(index >= sz)
                    index = 0;
                }
                set[index] = true;
                ++count;
            }
            for(auto i = 0U; i < sz; i++) {
                auto& n = m_neurons[i];
                if(set[i])
                    n.setActivationFunction(nullptr); // turn off
                else {
                    n.setActivationFunction(m_activationFunction);
                }
            }
        }
    }
    if(m_next)
        m_next->dropout(gen);
}

void Layer::inverseDropout(bool inherit) {
    if(!isOutput() && m_dropOut > 0.0) {
        const auto dropKeepRate =  1.0 / (1.0 - m_dropOut);
        for(auto& n : m_neurons) {
            n.setActivationFunction(m_activationFunction);
            for(auto i = 0U; i < n.size(); i++) {
                const auto weight = n.weight(i) * dropKeepRate;
                n.setWeight(i, weight);
            }
        }
        m_dropOut = 0.0;
    }
    if(m_next && inherit)
        m_next->inverseDropout();
}

void Layer::dropout(double dropoutRate, bool inherit) {
    neuropia_assert_always(dropoutRate >= 0.0 && dropoutRate < 1.0, "dropoutRate >= 0 && dropoutRate < 1.0");
    if(m_dropOut > 0.0) {
        inverseDropout(false);
    }
    m_dropOut = dropoutRate;
    if(inherit && m_next)
        m_next->dropout(dropoutRate, inherit);
}


Layer* Layer::get(int offset) {
    if(offset == 0)
        return this;
    else if(offset > 0) {
        if(m_next)
            return m_next->get(offset - 1);
    }
    else if(offset < 0) {
        if(m_prev)
            return m_prev->get(offset + 1);
    }
    return nullptr;
}

const Layer* Layer::get(int offset) const {
    return const_cast<Layer*>(this)->get(offset);
}


bool Layer::isValid() const {
    if(isInput())
        return true;
    const auto prevLayer = previousLayer(this);
    auto weights =
            Matrix<NeuronType>::fromArray(m_neurons,
                                      prevLayer->size(), //fully connected, amount of weights is prev layer neurons
    [](const Neuron & n, Matrix<NeuronType>::index_type index) {
        return n.weight(index);
    });
    return weights.reduce<bool>(false, [](bool a, auto r){return a || std::isnan(r) || std::isinf(r);});
}
