
#include "neuropia_simple.h"
#include <unordered_map>
#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "verify.h"
#include "default.h"

using namespace Neuropia;


#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
using namespace emscripten;
#endif

bool fatal(const char* t, const char* f, int line, const char* file) {
    std::cerr << "Assert:" << t << " in line " << line << " at " << f << "  in "<< file << "." << std::endl;
    std::abort();
}
#define ASSERT(X) ((X) || fatal("Invalid", __FUNCTION__, __LINE__, __FILE__))
#define ASSERT_X(X, T) ((X) || fatal((T), __FUNCTION__, __LINE__, __FILE__))

using namespace NeuropiaSimple;

template <size_t SZ>
class LogStream : public std::streambuf {
public:
    LogStream(std::function<void (const std::string&)>& logger) : m_logger(logger), m_os(this) {
        setp(m_buffer, m_buffer + SZ - 1);
    }

    ~LogStream() override {
    }
    /**
     * Since m_logger can be very very slow (on WASM) the frequent updates can be frozen
     */
    void freeze(bool doFreeze) {
        m_freeze = doFreeze;
    }

private:
    int_type overflow(int_type ch) override {
        if(ch != traits_type::eof()){
            *pptr() = static_cast<char>(ch);
            pbump(1);
            write();
        }
        return ch;
    }
    int sync() override {
        write();
        return 1;
    }
    void write() {
        const auto n = static_cast<size_t>(pptr() - pbase());
        if(!m_freeze) {
            const auto buf = std::string(m_buffer, n);
            m_logger(buf);
        }
        pbump(static_cast<int>(-n));
    }
private:
    std::function<void (const std::string&)> m_logger;
    char m_buffer[SZ];
    std::ostream m_os;
    bool m_freeze = false;
};

using SimpleLogStream = LogStream<2048>;

class SimpleTrainer : public TrainerBase {
public:
    SimpleTrainer(const std::string& root, const Neuropia::Params& params, bool quiet, const std::function<void (Layer&& layer, bool)>& onEnd) : TrainerBase(root, params, quiet), m_onEnd(onEnd) {}
    bool train(unsigned its, SimpleLogStream& stream);
    bool isOk() const {return isReady() && m_ok;}
private:
    bool train();
    std::function<void (Layer&& layer, bool)> m_onEnd;
    bool m_ok = true;
};

constexpr unsigned IoBufSz = 5;

class SimpleVerifier {
public:
    SimpleVerifier(const Neuropia::Layer& network, const std::string& imageFile, const std::string& labelFile) : m_network(network),
        m_testImages(imageFile, IoBufSz), m_testLabels(labelFile, IoBufSz) {
    }
    bool verify();
    NeuronType verifyResult() const {return m_testLabels.size() ? static_cast<NeuronType>(m_found) / static_cast<NeuronType>(m_testLabels.size()) : -1.0;}
private:
    const Neuropia::Layer& m_network;
    Neuropia::IdxReader<unsigned char> m_testImages;
    Neuropia::IdxReader<unsigned char> m_testLabels;
    size_t m_position = 0;
    unsigned m_found = 0;
};

NeuropiaPtr NeuropiaSimple::create(const std::string& root) {
    return std::make_shared<NeuropiaEnv>(root);
}


class NeuropiaSimple::NeuropiaEnv {
public:
    NeuropiaEnv(const std::string& root) :  m_root(root) {
        m_params.addHelp( topologyRe, "\',\'-separated list of integers");
        m_params.addHelp( activationFunctionRe, "\',\'-separated list of activation functions: \"sigmoid, relu or elu\"");
        m_params.addHelp( dropoutRateRe, "\',\'-separated of list of real numbers");
    }
    virtual ~NeuropiaEnv();
    void setLogger(std::function<void (const std::string&)> logger) {
        if(m_prevStreamBufCerr) {
            std::cerr.rdbuf(m_prevStreamBufCerr);
        }
        if(m_prevStreamBufCout) {
            std::cout.rdbuf(m_prevStreamBufCout);
        }
        m_logStream = std::make_unique<SimpleLogStream>(logger);
        m_prevStreamBufCout = std::cout.rdbuf(m_logStream.get());
        m_prevStreamBufCerr = std::cerr.rdbuf(m_logStream.get());
    }
    Layer m_network;
    Params m_params = {
        DEFAULT_PARAMS
};
    const std::string m_root;
    std::unique_ptr<SimpleLogStream> m_logStream;
    std::streambuf* m_prevStreamBufCout = nullptr;
    std::streambuf* m_prevStreamBufCerr = nullptr;
    bool m_once = false;

    std::unique_ptr<SimpleTrainer> m_trainer;
    std::unique_ptr<SimpleVerifier> m_verifier;
};

// to avoid warning
NeuropiaEnv::~NeuropiaEnv() {
    if(m_prevStreamBufCerr) {
        std::cerr.rdbuf(m_prevStreamBufCerr);
    }
    if(m_prevStreamBufCout) {
        std::cout.rdbuf(m_prevStreamBufCout);
    }
}



void NeuropiaSimple::free(NeuropiaPtr env) {
    env.reset();
}

std::vector<NeuronType> NeuropiaSimple::feed(NeuropiaPtr env, const std::vector<NeuronType>& input) {
    ASSERT(env && env->m_network.isValid());
    return env->m_network.feed(input);
}

bool NeuropiaSimple::isValid(const NeuropiaPtr& env, const std::string& name, const std::string& value) {
   ASSERT(env);
   return env->m_params.isValid(name, value);
}

bool NeuropiaSimple::setParam(const NeuropiaPtr& env, const std::string& name, const std::string& value) {
   ASSERT(env);
   return env->m_params.set(name, value);
}

// just make API more pleasant, the param is still changed to string... could be done in future that param is internally stored as a variant
// maybe more types could be introduced as int range and file path
bool NeuropiaSimple::setParam(const NeuropiaPtr& env, const std::string& name, const std::variant<int, double, float, bool>& value) {
    auto ok = false;
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T,bool>)
            ok = setParam(env, name, std::string(arg ? "true" : "false"));
        else 
            ok = setParam(env, name, std::to_string(arg));
    }, value);
    return ok;    
}

ParamType NeuropiaSimple::params(const NeuropiaPtr& env) {
    ASSERT(env);
    ParamType out;
    for(const auto& item : env->m_params) {
        const auto key = std::get<0>(item);
        if(key.length() > 0 && !(std::islower(key[0]))) {
            const std::vector<std::string>  data = {env->m_params.toType(std::get<2>(item)), std::get<1>(item), std::get<2>(item)};
            out.emplace(std::get<0>(item), data);
        }
    }
    return out;
}

class Once { //std::once makes exception, I dont want to
public:
    Once(bool& once) : m_once(once) {
        ASSERT(!m_once);
        m_once = true;
    }
    ~Once() {m_once = false;}
  private:
    bool& m_once;
};

bool NeuropiaSimple::train(const NeuropiaPtr& env, TrainType type) {
    ASSERT(env);
    if(env->m_once) {
        std::cerr << "Training is busy";
        return false;
    }
    Once once(env->m_once);
    std::unique_ptr<TrainerBase> trainer;
    switch (type) {
    case TrainType::Basic:
        trainer = std::make_unique<Trainer>(env->m_root, env->m_params, false);
        break;
    case TrainType::Evolutional:
        trainer = std::make_unique<TrainerEvo>(env->m_root, env->m_params, false);
        break;
    case TrainType::Parallel:
        trainer = std::make_unique<TrainerParallel>(env->m_root, env->m_params, false);
        break;
    }
    if(!trainer->isReady()) {
        std::cerr << "Training data is not ready" << std::endl;
        return false;
    }
    trainer->train();
    env->m_network = std::move(trainer->network());
   return true;
}

void NeuropiaSimple::save(const NeuropiaPtr& env, const std::string& filename, SaveType savetype) {
    ASSERT(env && env->m_network.isValid());
    const auto params = env->m_params.toMap();
    Neuropia::save(filename, env->m_network, params, savetype);
}

bool NeuropiaSimple::load(const NeuropiaPtr& env, const std::string& filename) {
    ASSERT(env);
    const auto loaded = Neuropia::load(Neuropia::absPath(env->m_root, filename));
    if(!loaded) {
        return false;
    }

    env->m_network = std::move(std::get<Layer>(*loaded));
    if(env->m_network.isValid()) {
        for(const auto& p : std::get<1>(*loaded)) {
            env->m_params.set(p.first, p.second);
        }
        return true;
    }
    return false;
}

int NeuropiaSimple::verify(const NeuropiaPtr& env) {
    ASSERT(env && env->m_network.isValid());
    const auto t = Neuropia::verify(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]),
                             Neuropia::absPath(env->m_root, env->m_params["LabelsVerify"]));
    return std::get<0>(t);
}

void NeuropiaSimple::setLogger(const NeuropiaPtr& env, std::function<void (const std::string&)> cb) {
    ASSERT(env);
    env->setLogger([cb](const std::string& str){
        cb(str);
    });
    std::cout << "Neuropia loaded" << std::endl;
}

bool SimpleTrainer::train(unsigned its, SimpleLogStream& stream) {
    const auto it = std::min(its, m_iterations);
    if(it <= m_iterations) {
        if(it == 0) {
            m_start = std::chrono::high_resolution_clock::now();
        }
        for(;;) {
            ++m_passedIterations;
            stream.freeze(m_passedIterations < it);
            const auto success = train();

            if(!success) {
                stream.freeze(false);
                train(); // to get error out
                m_onEnd(network(), false);
                return false;
            }

            if(m_passedIterations >= m_iterations) {
                stream.freeze(false);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << std::endl;
                std::cout << "Training "
                          << "timed:" << std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count()
                          << "." <<  std::chrono::duration_cast<std::chrono::microseconds>(stop - m_start).count()
                          - std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count() * 1000000 << std::endl;
                m_network.inverseDropout();
                m_onEnd(network(), isOk());
                break;
            }

            if(m_passedIterations >= it) {
                 stream.freeze(false);
                 break;
            }
        }
    }
    return true;
}


bool SimpleTrainer::train() {
    auto testVerify = m_testVerifyFrequency;

    if(m_maxTrainTime >= MaxTrainTime) {
        if(!m_quiet)
            persentage(m_passedIterations, m_iterations);
        m_learningRate += (1.0 / static_cast<NeuronType>(m_iterations)) * (m_learningRateMin - m_learningRateMax);
    } else {
        const auto stop = std::chrono::high_resolution_clock::now();
        const auto delta = static_cast<NeuronType>(std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count());
        if(delta > m_maxTrainTime) {
            return false;
        }
        const auto change = delta - m_gap;
        m_gap = delta;
        if(!m_quiet)
            persentage(delta, m_maxTrainTime, " " + std::to_string(m_learningRate));
        this->m_learningRate += (static_cast<NeuronType>(change) / static_cast<NeuronType>(m_maxTrainTime)) * (m_learningRateMin - m_learningRateMax);
    }

    const auto imageSize = m_images.size(1) * m_images.size(2);
    const auto at = m_random.random(m_images.size());
    const auto image = m_images.readAt(at, imageSize);
    const auto label = static_cast<unsigned>(m_labels.readAt(at));

#ifdef DEBUG_SHOW
    Neuropia::printimage(image.data()); //ASCII print images
    std::cout << label << std::endl;
#endif

    std::vector<Neuropia::NeuronType> inputs(imageSize);
    std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
    });

    std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
    outputs[label] = 1.0; //correct one is 1
    if(!this->m_network.train(inputs.begin(), outputs.begin(), m_learningRate, m_lambdaL2)) {
        m_ok = false;
        return false;
    }

    if(--testVerify == 0) {
        m_network.inverseDropout();
        printVerify(Neuropia::verify(m_network, m_imageFile, m_labelFile, 0, 200), "Test");
        setDropout();
        testVerify = m_testVerifyFrequency;
    }
    return true;
}

bool SimpleVerifier::verify() {
    const auto sz = m_testLabels.size();
    if(m_position >= sz)
        return false;
    const auto imageSize = m_testImages.size(1) * m_testImages.size(2);
    const auto image = m_testImages.read(imageSize);
    const auto label = static_cast<unsigned>(m_testLabels.read());

    std::vector<Neuropia::NeuronType> inputs(imageSize);

    std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
    });

    ASSERT_X(m_network.isValid(), std::string("On lap" + std::to_string(m_position)).c_str());

    const auto& outputs = m_network.feed(inputs.begin(), inputs.end());
    const auto max = static_cast<unsigned>(std::distance(outputs.begin(),
                                         std::max_element(outputs.begin(), outputs.end())));

    if(max == label) {
        ++m_found;
    }

    ++m_position;
    if((m_position % (sz / 10)) == 0) {
        Neuropia::persentage(m_position, sz);
    }

    return true;
}

bool train(const NeuropiaPtr& env, unsigned iteration) {
    ASSERT(env);

    if(iteration == 0 || !env->m_trainer) {
        env->m_logStream->freeze(false);
        std::cout << "Create trainer"  << std::endl;;
        env->m_trainer = std::make_unique<SimpleTrainer>(env->m_root, env->m_params, false, [env](Layer&& layer, bool){
            env->m_network = layer;
        });
    }

    return env->m_trainer->train(iteration, *env->m_logStream);
}

bool verify(const NeuropiaPtr& env, int iteration) {
    if(iteration == 0 || !env->m_verifier) {
        ASSERT(env->m_network.isValid());
        env->m_logStream->freeze(false);
        std::cout << "Create verifier"  << std::endl;;
        env->m_verifier.reset(new SimpleVerifier(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]), Neuropia::absPath(env->m_root,env->m_params["LabelsVerify"])));
        Neuropia::persentage(0, 1);
    }
    return env->m_verifier->verify();
}


ParamType basicParams(const NeuropiaPtr& env) {
    auto p = NeuropiaSimple::params(env);
    p.erase("Jobs");    //mt
    p.erase("BatchSize"); //mt
    p.erase("BatchVerifySize"); //mt
    p.erase("Hard"); //for ensenble
    p.erase("Extra"); //this is for ensemble
    p.erase("LearningRate"); //use only min and max
    p.erase("Classes"); //irrelevant
    return p;
    return p;
}

bool isNetworkValid(const NeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_trainer) {
        env->m_logStream->freeze(false);
    }
    return env->m_network.isValid();
}

NeuronType verifyResult(const NeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_verifier) {
        env->m_logStream->freeze(false);
        return env->m_verifier->verifyResult();
    }
    return -2.0; //arbitrary negative number, but not -1
}

int showImage(const NeuropiaPtr& env, const std::string& imageName, const std::string& labelName, unsigned index) {
    ASSERT(env);
    Neuropia::IdxReader<std::array<unsigned char, 28 * 28>> idxi(Neuropia::absPath(env->m_root, imageName));
    if(!idxi.ok() || idxi.dimensions() != 3)
        return -1;

    Neuropia::IdxReader<unsigned char> idxl(Neuropia::absPath(env->m_root, labelName));
    if(!idxl.ok() || idxl.dimensions() != 1)
        return -2;

    const auto data = idxi.readAt(index);
    Neuropia::printimage(data.data());
    return idxl.readAt(index);
}


#ifdef __EMSCRIPTEN__
void setLogger(const NeuropiaPtr& env, emscripten::val cb) {
    NeuropiaSimple::setLogger(env, [cb](const std::string& str){
        cb(str);
    });
}


template <typename T>
std::vector<T> fromJSArray(const emscripten::val& v) {
    const auto l = v["length"].as<unsigned>();
    std::vector<T> vec(l);
    emscripten::val memoryView(emscripten::typed_memory_view(l, vec.data()));
    memoryView.call<void>("set", v);
    return vec;
}

std::vector<NeuronType> feed(NeuropiaPtr env, emscripten::val a) {
    const auto image = fromJSArray<NeuronType>(a);

    std::vector<unsigned char> test(image.size());
    std::transform(image.begin(), image.end(), test.begin(), [](NeuronType c) {
        ASSERT(c >=  0 && c <= 255);
        return static_cast<unsigned char>(c);
    });
/* debug the map
    Neuropia::printimage(test.data());
    std::cout << std::endl;

    std::cout << "reference" << std::endl;
    std::cout << showImage(env, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", 3) << std::endl;
*/

    std::vector<Neuropia::NeuronType> inputs(image.size());
    std::transform(image.begin(), image.end(), inputs.begin(), [](NeuronType c) {
        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0., 255.);
    });
    return NeuropiaSimple::feed(env, inputs);
}


#endif




#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<NeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<NeuropiaEnv, const std::string&>);
    register_vector<NeuronType>("ValueVector");
    register_vector<std::string>("StringVector");
    register_map<ParamType::key_type, ParamType::mapped_type>("ParamMap");
    function("create", &NeuropiaSimple::create);
    function("free", &NeuropiaSimple::free);
    function("feed", &::feed);
    function("setParam", &NeuropiaSimple::setParam);
    function("params", &::basicParams);
    function("train", &::train);
    function("save", &NeuropiaSimple::save);
    function("load", &NeuropiaSimple::load);
    function("verify", &::verify);
    function("setLogger", &::setLogger);
    function("isNetworkValid", &::isNetworkValid);
    function("verifyResult", &::verifyResult);
    function("showImage", &::showImage);
}
#endif

