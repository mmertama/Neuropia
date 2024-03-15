#include "neuropia.h"
#include <random>
#include <iostream>
#include <cstring>
#include <optional>
#include "matrix.h"

using namespace Neuropia;

#ifdef NO_VERBOSE
#define print_error(x) 
#else
#define print_error(x) std::cerr << "Error: " << x << std::endl 
#endif


template<typename T>
void write(std::ofstream& strm, const T& value) {
    strm.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template<>
void write(std::ofstream& strm, const std::string& value) {
    write<uint8_t>(strm, static_cast<uint8_t>(value.length()));
    strm.write(reinterpret_cast<const char*>(value.data()), static_cast<signed>(value.length()));
}


class Neuropia::StreamBase  {
public:
    template<typename T>
    std::optional<T> read() {
        T v;
        constexpr auto sz{sizeof(T)};
        return sz == read(reinterpret_cast<char*>(&v), sz) ? std::make_optional(v) : std::nullopt;
    }


    std::optional<std::string> read_string(size_t sz) {
        std::string str (sz, '\0');
        return sz == read(str.data(), sz) ? std::make_optional(str) : std::nullopt;
    }

    template <typename T>
    std::optional<std::string> read_value() {
        const auto len = read<T>();
        if(!len) {
            print_error("Read error! when reading " + std::to_string(sizeof(T)));
            return std::nullopt;
        }
        const auto val = read_string(*len);
        return val;
    }
    virtual size_t read(char* target, size_t size) = 0; 
    virtual bool eof() const = 0; 
    virtual ~StreamBase() = default;
};

class ByteStream : public StreamBase {
public:
    ByteStream(const std::vector<uint8_t>& vec) : m_vec(vec) {}
    size_t read(char* target, size_t size) {
        auto sz = std::min(m_vec.size() - m_pos, size);
        if(sz > 0) {
            std::memcpy(target, &m_vec[m_pos], sz);
            m_pos += sz;
        }
        return sz;
    }
    bool eof() const {return m_pos >= m_vec.size();}
private:
    const std::vector<uint8_t>& m_vec;
    size_t m_pos = 0;
};

class IfStream : public StreamBase {
public:
    IfStream(std::ifstream& strm) : m_strm{strm} {
        neuropia_assert_always(strm.is_open() && strm.good(), "File is not open");
    }
    size_t read(char* target, size_t size) {
        m_strm.read(target, static_cast<std::streamsize>(size));
        const auto gpos = static_cast<std::size_t>(m_strm.gcount());
        return gpos;
        }
    bool eof() const {return m_strm.eof();}    
private:
    std::ifstream& m_strm;
};

class BytePtrStream : public StreamBase {
public:
    BytePtrStream(const uint8_t* bytes, size_t sz) : m_bytes(bytes), m_sz(sz) {}
    size_t read(char* target, size_t size) {
        auto sz = std::min(m_sz - m_pos, size);
        if(sz > 0) {
            std::memcpy(target, &m_bytes[m_pos], sz);
            m_pos += sz;
        }
        return sz;
    }
    bool eof() const {return m_pos >= m_sz;}
    BytePtrStream(const BytePtrStream&) = delete;
    BytePtrStream& operator=(const BytePtrStream&) = delete;
private:
    const uint8_t* m_bytes;
    const size_t m_sz;
    size_t m_pos = 0;
};


static
std::optional<MetaInfo> readMeta(StreamBase& strm) {
    std::unordered_map<std::string, std::string> map;
    const auto sz = strm.read<uint8_t>();
    for(auto i = 0; i < sz; i++) {
        const auto key = strm.read_value<uint8_t>();
        if(!key) return std::nullopt;
        const auto val = strm.read_value<uint8_t>();
        if(!val) return std::nullopt;
        map[*key] = *val;
    }
    return map;
}

static void writeMeta(std::ofstream& strm, const std::unordered_map<std::string, std::string>& map ) {
    const auto sz = static_cast<uint8_t>(map.size());
    write(strm, sz);
    for(const auto& p : map) {
        write(strm, p.first);
        write(strm, p.second);
    }
}

std::ostream& operator << (std::ostream& strm, const std::vector<Neuropia::NeuronType>& values) {
    strm << '[';
    if(!values.empty()) {
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

template <typename T> 
void write_neuronType(std::ofstream& stream, NeuronType nt) {
    const auto v = static_cast<T>(nt);
    stream.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

void Neuron::save(std::ofstream& stream, SaveType saveType) const {

    std::function<void (std::ofstream& stream, NeuronType n)> write_fn = nullptr;

    switch (saveType) {
    case SaveType::SameAsNeuronType:
        write_fn = &write_neuronType<NeuronType>;
        break;
    case SaveType::Double:
        write_fn = &write_neuronType<double>;
        break;
    case SaveType::Float:
        write_fn = &write_neuronType<float>;
        break;
    case SaveType::LongDouble:
        write_fn = &write_neuronType<long double>;
        break;        
    default:
        neuropia_assert_always(false, "bad");    
    }


    const auto sz = static_cast<std::uint32_t>(m_weights.size());
    write(stream, sz);
    for(const auto& w : m_weights) {
        write_fn(stream, w);
    }
    write_fn(stream, m_bias);
}


bool Neuron::load(std::ifstream& stream, SaveType saveType) {
    IfStream if_stream{stream};
    return loadNeuron(if_stream, saveType);
}

bool Neuron::load(const std::vector<uint8_t>& data, SaveType saveType) {
    ByteStream stream{data};
    return loadNeuron(stream, saveType);
}

bool Neuron::loadNeuron(StreamBase& stream, SaveType saveType) {
    m_weights.clear();
    const auto size = stream.read<uint32_t>();
    if(!size || stream.eof()) {
        print_error("Invalid neuron");
        return false;
    }
    m_weights.resize(*size);
    auto neuron_sz = sizeof(NeuronType);
    switch (saveType) {
    case SaveType::Float:
        neuron_sz = sizeof(float);
        break;
    case SaveType::Double:
        neuron_sz = sizeof(double); 
        break;
    case SaveType::LongDouble: 
        neuron_sz = sizeof(long double); 
        break;
    case SaveType::SameAsNeuronType: 
        neuron_sz = sizeof(NeuronType); 
        break;
    default:
        neuropia_assert_always(false, "bad");        
    } 
    for(size_t s = 0; s < *size; s++) {
        NeuronType w{};
        if(neuron_sz != stream.read(reinterpret_cast<char*>(&w), neuron_sz))
            return false;
        setWeight(s, w);
    }
    if(stream.eof()) {
        print_error("Corrupted neuron");
        return false;
    }
    NeuronType b;
    if(neuron_sz != stream.read(reinterpret_cast<char*>(&b), neuron_sz)) {
        print_error("Cannot read bias");
        return false;
    }
    setBias(b);
    return true;
}

size_t Neuron::consumption() const {
    return sizeof(this) + m_weights.size() * sizeof(decltype(m_weights)::value_type);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////7


Layer::Layer(const std::initializer_list<Neuron>& list, const ActivationFunction& activationFunction) noexcept : m_neurons(list), m_activationFunction(activationFunction), m_outBuffer(list.size()) {
}

Layer::Layer(size_t count, const ActivationFunction& activationFunction, const Neuron& proto) noexcept : m_activationFunction(activationFunction) {
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

const Layer* Layer::outLayer() const {
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
bool Layer::backpropagation(const ValueVector& outValues, const ValueVector& expectedValues, NeuronType learningRate, NeuronType lambdaL2, const DerivativeFunction& df) {

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
            const auto L2 = gradients.reduce<NeuronType>(0, [](auto a, auto r){return a + (r * r);}) /
                static_cast<NeuronType>(gradients.rows());
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

constexpr char H4[] = {'N', 'E', 'U', '0', '0', '0', '0', '4'};
//constexpr char H2[] = {'N', 'E', 'U', '0', '0', '0', '0', '2'};
//constexpr char H3[] = {'N', 'E', 'U', '0', '0', '0', '0', '3'};

static bool Hcomp(const std::string& h, const char* H = H4) {
    for(auto i = 0U; i < sizeof(H); i++)
        if(h[i] != H[i]) {
            return false;
        }
    return true;
}

void Layer::save(std::ofstream& strm, const std::unordered_map<std::string, std::string>& meta, SaveType saveType) const {
    if(isInput()) {
        write(strm, H4);
        write(strm, saveType);
        unsigned layers = 1;
        auto* layer = this;
        while(layer->m_next) {
            ++layers;
            layer = layer->m_next.get();
        }
        write<uint8_t>(strm, static_cast<uint8_t>(layers));
        writeMeta(strm, meta);
    }

    write(strm, m_activationFunction.name());

    const auto  dropout =  static_cast<std::uint32_t>(m_dropOut * 100000.);
    write(strm, dropout);

    const auto sz = static_cast<std::uint32_t >(m_neurons.size());
    write(strm, sz);

    for(const auto& n : m_neurons) {
        n.save(strm, saveType);
    }

    if(m_next) {
        m_next->save(strm, {}, saveType);
    }
}

static
std::optional<Header> read_header(StreamBase& stream) {
    const auto hdr =  stream.read_string(sizeof(H4));
    if(hdr) {
        if(Hcomp(*hdr)) {
            const auto save_type = stream.read<uint8_t>();
            const auto layer_count = stream.read<uint8_t>();

            if(layer_count && save_type && *save_type <= static_cast<uint8_t>(SaveType::LongDouble) ) {
                return std::make_optional(Header{static_cast<SaveType>(*save_type), *layer_count});
            }
        }
#if 0 // no more comp
         if(Hcomp(*hdr, H3)) {
            const auto save_type = stream.read<uint8_t>();
            if(save_type && *save_type <= static_cast<uint8_t>(SaveType::LongDouble) ) {
                return std::make_optional(static_cast<SaveType>(*save_type));
            }
        }
        if(Hcomp(*hdr, H2)) {
            return {SaveType::SameAsNeuronType};
        }
#endif        
    }    
    print_error("Corrupted import stream or wrong version");
    return std::nullopt;
    }

std::optional<MetaInfo> Layer::doLoad(StreamBase& strm) {
    const auto header = read_header(strm);
    if(!header) {
        print_error("bad header");
        return std::nullopt;
    }

    const auto [save_type, layer_count] = header.value();

    const auto meta = readMeta(strm);
    if(!meta) {
        print_error("bad meta");
        return std::nullopt;
    }

    if(layer_count == 0 || !loadLayer(strm, save_type, layer_count - 1)) {
        print_error("invalid network");
        return std::nullopt;
    }
    return meta;
}

std::optional<MetaInfo> Layer::load(std::ifstream& if_strm)  {
    IfStream strm(if_strm);
    return doLoad(strm);
    }

std::optional<MetaInfo> Layer::load(const std::vector<uint8_t>& data) {
    ByteStream strm(data);
    return doLoad(strm);
    }

std::optional<MetaInfo> Layer::load(const uint8_t* bytes, size_t sz) {
    BytePtrStream strm(bytes, sz);
    return doLoad(strm);
    }


bool Layer::loadLayer(StreamBase &strm, SaveType saveType, unsigned layer_index) {
    const auto name = strm.read_value<uint8_t>();
    if(!name) {
        print_error("Cannot read activation function");
        return false;
    }

    if(signumFunction.name() == *name)
        m_activationFunction = signumFunction;
    else if(binaryFunction.name() == *name)
        m_activationFunction = binaryFunction;
    else if(sigmoidFunction.name() == *name)
        m_activationFunction = sigmoidFunction;
    else if(reLuFunction.name() == *name)
        m_activationFunction = reLuFunction;
    else if(eluFunction.name() == *name)
        m_activationFunction = eluFunction;
    else {
        print_error("Invalid activation function name " + *name);
        return false;
    }    

    const auto dropout = strm.read<uint32_t>();
    if(!dropout) {
        print_error("Cannot read dropout");
        return false;
    }

    m_dropOut = static_cast<NeuronType>(*dropout) / 100000.;

    const auto count = strm.read<uint32_t>();
    if(!count) {
        print_error("Cannot read count");
        return false;
    }

   fill(*count, Neuron(m_activationFunction));

    for(auto& n : m_neurons) {
        if(!n.loadNeuron(strm, saveType))
            return false;
       }

    if(layer_index > 0) {
        auto layer = new Layer();
        if(!layer->loadLayer(strm, saveType, layer_index - 1))
            return false;
        neuropia_assert_always(layer->size() > 0, "Invalid layer"); 
        m_next.reset(layer);
        m_next->m_prev = this;
    }

    return (layer_index > 0 || !strm.read<uint8_t>());
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
        m_next = std::make_unique<Layer>(*other.m_next);
    }
    if(m_next) {
        m_next->m_prev = this;
    }
    return *this;
}

void Layer::merge(const Layer& other, NeuronType factor) {
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
        NeuronType r = 0;
        switch (strategy) {
        case Layer::InitStrategy::Norm:
            r = 1.0;
            break;
        case Layer::InitStrategy::Logistic:
             r = std::sqrt(6.0 / static_cast<double>(m_neurons.size() + m_prev->m_neurons.size()));
            break;
        case Layer::InitStrategy::ReLu:
             r = std::sqrt(2.0) *  std::sqrt(6.0 / static_cast<double>(m_neurons.size() + m_prev->m_neurons.size()));
            break;
        default:
            neuropia_assert_always(false, "bad");        
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
            const auto sz = static_cast<unsigned>(m_neurons.size());
            const auto dropCount = static_cast<unsigned>(static_cast<NeuronType>(sz * m_dropOut));
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

void Layer::dropout(NeuronType dropoutRate, bool inherit) {
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

    if(offset > 0) {
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


bool Layer::isValid(bool testNext) const {
    if(m_outBuffer.empty()) {
        return false;
    }
    if(isInput())
        return !testNext || !m_next || m_next->isValid(true);
    const auto prevLayer = previousLayer(this);
    auto weights =
            Matrix<NeuronType>::fromArray(m_neurons,
                                      prevLayer->size(), //fully connected, amount of weights is prev layer neurons
    [](const Neuron & n, Matrix<NeuronType>::index_type index) {
        return n.weight(index);
    });
    const auto hasInvalidValue = weights.reduce<bool>(false, [](bool a, auto r){
           return a || std::isnan(r) || std::isinf(r);
       });
    return !hasInvalidValue && (!testNext || !m_next || m_next->isValid(true));
}

const Layer* Layer::inputLayer() const {
    auto input = this;
    for(;;) {
        const auto prev = previousLayer(input);
        return input;
        input = prev;    
    }
}

Sizes Layer::sizes() const {
    return Sizes{
                static_cast<unsigned>(inputLayer()->size()),
                static_cast<unsigned>(outLayer()->size())
                };
}

size_t Layer::consumption(bool cumulative) const {
    const auto c = sizeof(this) 
    + m_outBuffer.size() * sizeof(decltype(m_outBuffer)::value_type)
    + std::accumulate(m_neurons.begin(), m_neurons.end(), 0U, [](const auto& a, const auto& n ) {
        return a + n.consumption();
    });

    return cumulative && m_next ? c + m_next->consumption(cumulative) : c;
}

std::optional<Header> Neuropia::isValidFile(const std::string& filename) {
    std::ifstream is;
    is.open(filename, std::ios::binary);
    if(!is.is_open())
        return std::nullopt;
    IfStream i(is);
    return read_header(i);
}