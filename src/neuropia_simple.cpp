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

class NeuropiaSimple::NeuropiaEnv {
public:
    NeuropiaEnv(const std::string& root) :  m_root(root) {}
    Layer m_network;
    Params m_params = {
    {"ImagesVerify", "", Neuropia::Params::String},
    {"LabelsVerify", "", Neuropia::Params::String},
    {"Images", "", Neuropia::Params::String},
    {"Labels", "", Neuropia::Params::String},
    {"Iterations", "1", Neuropia::Params::Int},
    {"Jobs", "1", Neuropia::Params::Int},
    {"LearningRate", "0", Neuropia::Params::Real},
    {"LearningRateMin", "0.02", Neuropia::Params::Real},
    {"LearningRateMax", "0.02", Neuropia::Params::Real},
    {"BatchSize", "800", Neuropia::Params::Int},
    {"BatchVerifySize", "100", Neuropia::Params::Int},
    {"Topology", "64,32", topologyRe},
    {"MaxTrainTime", std::to_string(MaxTrainTime), Neuropia::Params::Int},
    {"File", "mnistdata.bin", Neuropia::Params::String},
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

bool NeuropiaSimple::setParam(NeuropiaPtr env, const std::string& name, const std::string& value) {
   ASSERT(env);
   return env->m_params.set(name, value);
}

std::map<std::string, std::string> NeuropiaSimple::params(NeuropiaPtr env) {
    ASSERT(env);
    std::map<std::string, std::string> out;
    for(const auto& item : env->m_params) {
        out.emplace(std::get<0>(item), std::get<1>(item));
    }
    return out;
}

bool NeuropiaSimple::train(NeuropiaPtr env, TrainType type) {
    ASSERT(env);
    std::unique_ptr<TrainerBase> trainer;
    switch (type) {
        case TrainType::Basic:
            trainer = std::make_unique<Trainer>(env->m_root, env->m_params, true);
        break;
    case TrainType::Evolutional:
        trainer = std::make_unique<TrainerEvo>(env->m_root, env->m_params, true);
        break;
    case TrainType::Parallel:
        trainer = std::make_unique<TrainerParallel>(env->m_root, env->m_params, true);
        break;
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
EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<NeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<NeuropiaEnv, const std::string&>);
    register_vector<double>("ValueVector");
    enum_<TrainType>("TrainType")
            .value("BASIC", TrainType::Basic)
            .value("EVOLUTIONAL", TrainType::Evolutional)
            .value("PARALLEL", TrainType::Parallel)
            ;
    register_map<std::string, std::string>("map<string, string>");
    function("create", &NeuropiaSimple::create);
    function("free", &NeuropiaSimple::free);
    function("feed", &NeuropiaSimple::feed);
    function("setParam", &NeuropiaSimple::setParam);
    function("params", &NeuropiaSimple::params);
    function("train", &NeuropiaSimple::train);
    function("save", &NeuropiaSimple::save);
    function("load", &NeuropiaSimple::load);
    function("verify", &NeuropiaSimple::verify);
}
#endif

