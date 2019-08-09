
#include <cstring>
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

bool fatal(const char* t, const char* f, int line) {
    std::cerr << "Assert:" << t << " in line " << line << " at " << f << "." << std::endl;
    return false;
}
#define ASSERT(X) (X || fatal("Invalid", __FUNCTION__, __LINE__))
#define ASSERT_X(X, T) (X || fatal(T, __FUNCTION__, __LINE__))

using namespace NeuropiaSimple;

constexpr char topologyRe[] = R"(\d+(,\d+)*$)";
constexpr char activationFunctionRe[] =R"((sigmoid|relu|elu)(,(sigmoid|relu|elu))*$)";
constexpr char dropoutRateRe[] = R"(\d+\.?\d*(,\d+\.?\d*)*$)";

template <size_t SZ>
class LogStream : public std::streambuf {
public:
    LogStream(std::function<void (const std::string&)> logger) : m_logger(logger), m_os(this) {
        setp(m_buffer, m_buffer + SZ - 1);
    }
    ~LogStream() override {
    }
    std::ostream& stream() {
        return m_os;
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
        std::ptrdiff_t n = pptr() - pbase();

        char ntBuf[SZ];
        std::memcpy(ntBuf, m_buffer, static_cast<size_t>(n));
        ntBuf[n] = '\0';
        m_logger(ntBuf);
        pbump(static_cast<int>(-n));
    }
private:
    std::function<void (const std::string&)> m_logger;
    char m_buffer[SZ];
    std::ostream m_os;
};

class NeuropiaSimple::NeuropiaEnv {
public:
    NeuropiaEnv(const std::string& root) :  m_root(root) {
    }
    virtual ~NeuropiaEnv() {
        if(m_prevStreamBufCerr) {
            std::cerr.rdbuf(m_prevStreamBufCerr);
        }
        if(m_prevStreamBufCout) {
            std::cout.rdbuf(m_prevStreamBufCout);
        }
    }
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
    {"Iterations", "1", Neuropia::Params::Int},
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
};

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

bool NeuropiaSimple::train(NeuropiaPtr env, TrainType type) {
    ASSERT(env);
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
    return env->m_network.isValid();
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

#ifdef __EMSCRIPTEN__
void setLogger(NeuropiaPtr env, emscripten::val cb) {
    ASSERT(env);
    env->setLogger([cb](const std::string& str){
        cb(str);
    });
    std::cout << "logging started" << std::endl;
}
#endif

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<NeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<NeuropiaEnv, const std::string&>);
    register_vector<double>("ValueVector");
    register_vector<std::string>("StringVector");
    enum_<TrainType>("TrainType")
            .value("BASIC", TrainType::Basic)
            .value("EVOLUTIONAL", TrainType::Evolutional)
            .value("PARALLEL", TrainType::Parallel)
            ;
    register_map<ParamType::key_type, ParamType::mapped_type>("ParamMap");
    function("create", &NeuropiaSimple::create);
    function("free", &NeuropiaSimple::free);
    function("feed", &NeuropiaSimple::feed);
    function("setParam", &NeuropiaSimple::setParam);
    function("params", &NeuropiaSimple::params);
    function("train", &NeuropiaSimple::train);
    function("save", &NeuropiaSimple::save);
    function("load", &NeuropiaSimple::load);
    function("verify", &NeuropiaSimple::verify);
    function("setLogger", &setLogger);
}
#endif

