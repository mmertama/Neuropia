
#include "neuropia_simple.h"
#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "verify.h"

using namespace Neuropia;


#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
using namespace emscripten;
#endif

bool fatal(const char* t, const char* f, int line, const char* file) {
    std::cerr << "Assert:" << t << " in line " << line << " at " << f << "  in "<< file << "." << std::endl;
    return false;
}
#define ASSERT(X) (X || fatal("Invalid", __FUNCTION__, __LINE__, __FILE__))
#define ASSERT_X(X, T) (X || fatal(T, __FUNCTION__, __LINE__, __FILE__))

using namespace NeuropiaSimple;

constexpr char topologyRe[] = R"(\d+(,\d+)*$)";
constexpr char activationFunctionRe[] =R"((sigmoid|relu|elu)(,(sigmoid|relu|elu))*$)";
constexpr char dropoutRateRe[] = R"(\d+\.?\d*(,\d+\.?\d*)*$)";

std::string escaped(const std::string& data) {
    std::string escaped;
    for (const auto& c : data) {
        if(c < ' ')
            continue;
        switch (c) {
        case '\"': escaped  += "&quot;"; break;
        case '&':  escaped  += "&amp;";  break;
        case '<':  escaped  +=  "&lt;";   break;
        case '>':  escaped  += "&gt;";   break;
        default:   escaped  +=  c;
        }
    };
    return escaped;
}

class SimpleTrainer : public TrainerBase
{
public:
    SimpleTrainer(const std::string& root, const Neuropia::Params& params, bool quiet) : TrainerBase(root, params, quiet) {}
    bool train(size_t it) {
        bool success = true;
        if(it < m_iterations) {
            while(success && m_passedIterations < it) {
                ++m_passedIterations;
                success = train();
                if(m_passedIterations + 1 == m_iterations && success) {
                    m_network.inverseDropout();
                    break;
                }
            }
        } else success = false;
        return success;
    }

    bool isOk() const {return isReady() && m_ok;}
private:
    bool train();
    bool m_ok = true;
};

template <size_t SZ>
class LogStream : public std::streambuf {
public:
    LogStream(std::function<void (const std::string&)> logger) : m_logger(logger), m_os(this) {
        setp(m_buffer, m_buffer + SZ - 1);
    }

    ~LogStream() override {
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
        const auto buf = std::string(m_buffer, n);
        m_logger(buf);
        pbump(static_cast<int>(-n));
    }
private:
    std::function<void (const std::string&)> m_logger;
    char m_buffer[SZ];
    std::ostream m_os;
};

class NeuropiaSimple::NeuropiaEnv {
public:
    NeuropiaEnv(const std::string& root) :  m_root(root) {}
    virtual ~NeuropiaEnv();
    void setLogger(std::function<void (const std::string&)> logger) {
        if(m_prevStreamBufCerr) {
            std::cerr.rdbuf(m_prevStreamBufCerr);
        }
        if(m_prevStreamBufCout) {
            std::cout.rdbuf(m_prevStreamBufCout);
        }
        m_logstream.reset(new LogStream<1024>(logger));
        m_prevStreamBufCout = std::cout.rdbuf(m_logstream.get());
        m_prevStreamBufCerr = std::cerr.rdbuf(m_logstream.get());
    }
    Layer m_network;
    Params m_params = {
    {"ImagesVerify", "", Neuropia::Params::File},
    {"LabelsVerify", "", Neuropia::Params::File},
    {"Images", "", Neuropia::Params::File},
    {"Labels", "", Neuropia::Params::File},
    {"Iterations", "1000", Neuropia::Params::Int},
    {"Jobs", "1", Neuropia::Params::Int},
    {"LearningRate", "0", Neuropia::Params::Real},
    {"LearningRateMin", "0.02", Neuropia::Params::Real},
    {"LearningRateMax", "0.02", Neuropia::Params::Real},
    {"BatchSize", "800", Neuropia::Params::Int},
    {"BatchVerifySize", "100", Neuropia::Params::Int},
    {"Topology", "64,32", topologyRe},
    {"MaxTrainTime", std::to_string(static_cast<int>(MaxTrainTime)), Neuropia::Params::Int},
    {"File", "mnistdata.bin", Neuropia::Params::File},
    {"Extra", "", Neuropia::Params::String},
    {"Hard", "false", Neuropia::Params::Bool},
    {"ActivationFunction", "sigmoid", activationFunctionRe},
    {"InitStrategy", "auto", R"((auto|logistic|norm|relu)$)"},
    {"DropoutRate", "0.0", dropoutRateRe},
    {"TestFrequency", "9999999", Neuropia::Params::Int},
    {"L2", "0.0", Neuropia::Params::Real},
    {"Classes", "10", Neuropia::Params::Int}
};
    const std::string m_root;
    std::unique_ptr<LogStream<1024>> m_logstream;
    std::streambuf* m_prevStreamBufCout = nullptr;
    std::streambuf* m_prevStreamBufCerr = nullptr;
    bool m_once = false;

    std::unique_ptr<SimpleTrainer> m_trainer;
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

NeuropiaPtr NeuropiaSimple::create(const std::string& root) {
    return std::make_shared<NeuropiaEnv>(root);
}


void NeuropiaSimple::free(NeuropiaPtr env) {
    env.reset();
}

std::vector<double> NeuropiaSimple::feed(NeuropiaPtr env, const std::vector<double>& input) {
    ASSERT(env && env->m_network.isValid());
    return env->m_network.feed(input);
}

bool NeuropiaSimple::isValid(NeuropiaPtr env, const std::string& name, const std::string& value) {
   ASSERT(env);
   return env->m_params.isValid(name, value);
}

bool NeuropiaSimple::setParam(NeuropiaPtr env, const std::string& name, const std::string& value) {
   ASSERT(env);
   return env->m_params.set(name, value);
}

ParamType NeuropiaSimple::params(NeuropiaPtr env) {
    ASSERT(env);
    ParamType out;
    for(const auto& item : env->m_params) {
        const auto key = std::get<0>(item);
        if(key.length() > 0 && !(std::islower(key[0]))) {
            const std::vector<std::string>  data = {env->m_params.toType(std::get<2>(item)), std::get<1>(item)};
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


bool NeuropiaSimple::train(NeuropiaPtr env, TrainType type) {
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
   return false;
}

void NeuropiaSimple::save(NeuropiaPtr env, const std::string& filename) {
    ASSERT(env && env->m_network.isValid());
    Neuropia::save(filename, env->m_network);
}

bool NeuropiaSimple::load(NeuropiaPtr env, const std::string& filename) {
    ASSERT(env);
    const auto nets = Neuropia::load(filename);
    if(nets.size() > 0) {
        env->m_network = nets[0];
        return env->m_network.isValid();
    }
    return false;
}

int NeuropiaSimple::verify(NeuropiaPtr env) {
    ASSERT(env && env->m_network.isValid());
    const auto t = Neuropia::verify(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]),
                             Neuropia::absPath(env->m_root, env->m_params["LabelsVerify"]));
    return std::get<0>(t);
}

void NeuropiaSimple::setLogger(NeuropiaPtr env, std::function<void (const std::string&)> cb) {
    ASSERT(env);
    env->setLogger([cb](const std::string& str){
        cb(str);
    });
    std::cout << "logging started" << std::endl;
}


bool SimpleTrainer::train() {
    auto testVerify = m_testVerifyFrequency;

    if(m_maxTrainTime >= MaxTrainTime) {
        if(!m_quiet)
            persentage(m_passedIterations, m_iterations);
        m_learningRate += (1.0 / static_cast<double>(m_iterations)) * (m_learningRateMin - m_learningRateMax);
    } else {
        const auto stop = std::chrono::high_resolution_clock::now();
        const auto delta = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count());
        if(delta > m_maxTrainTime) {
            return false;
        }
        const auto change = delta - m_gap;
        m_gap = delta;
        if(!m_quiet)
            persentage(delta, m_maxTrainTime, " " + std::to_string(m_learningRate));
        this->m_learningRate += (static_cast<double>(change) / static_cast<double>(m_maxTrainTime)) * (m_learningRateMin - m_learningRateMax);
    }

    const auto imageSize = m_images.size(1) * m_images.size(2);
    const auto at = m_images.random();
    const auto image = m_images.next(at, imageSize);
    const auto label = static_cast<unsigned>(m_labels.next(at));

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




#ifdef __EMSCRIPTEN__
void setLogger(NeuropiaPtr env, emscripten::val cb) {
    NeuropiaSimple::setLogger(env, [cb](const std::string& str){
        cb(str);
    });
}


bool train(NeuropiaPtr env, int iteration) {
    ASSERT(env);

    if(iteration == 0 || !env->m_trainer) {
        std::cout << "Create trainer\n" << std::endl;
        env->m_trainer.reset(new SimpleTrainer(env->m_root, env->m_params, false));
    }
    return env->m_trainer->train(iteration);
}

ParamType basicParams(NeuropiaPtr env) {
    auto p = NeuropiaSimple::params(env);
    p.erase("Jobs");
    p.erase("BatchSize");
    p.erase("BatchVerifySize");
    p.erase("Hard");
    return p;
}

bool trainingOk(NeuropiaPtr env) {
    ASSERT(env);
    return env->m_trainer && env->m_trainer->isOk();
}

#endif

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<NeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<NeuropiaEnv, const std::string&>);
    register_vector<double>("ValueVector");
    register_vector<std::string>("StringVector");
    register_map<ParamType::key_type, ParamType::mapped_type>("ParamMap");
    function("create", &NeuropiaSimple::create);
    function("free", &NeuropiaSimple::free);
    function("feed", &NeuropiaSimple::feed);
    function("setParam", &NeuropiaSimple::setParam);
    function("params", &::basicParams);
    function("train", &::train);
    function("save", &NeuropiaSimple::save);
    function("load", &NeuropiaSimple::load);
    function("verify", &NeuropiaSimple::verify);
    function("setLogger", &::setLogger);
    function("trainingOk", &::trainingOk);
}
#endif

