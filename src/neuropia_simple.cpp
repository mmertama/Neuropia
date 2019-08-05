#include "neuropia_simple.h"
#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "verify.h"

bool fatal(const char* t, const char* f, int line) {
    std::cerr << "Assert:" << t << " in line " << line << " at " << f << "." << std::endl;
    return false;
}
#define ASSERT(X) (X || fatal("Invalid", __FUNCTION__, __LINE__))
#define ASSERT_X(X, T) (X || fatal(T, __FUNCTION__, __LINE__))

using namespace Neuropia;

constexpr char topologyRe[] = R"(\d+(,\d+)*$)";
constexpr char activationFunctionRe[] =R"((sigmoid|relu|elu)(,(sigmoid|relu|elu))*$)";
constexpr char dropoutRateRe[] = R"(\d+\.?\d*(,\d+\.?\d*)*$)";

class NeuropiaEnv {
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

NeuropiaEnv* createNeuropia(const std::string& root) {
    return new NeuropiaEnv(root);
}

void free(NeuropiaEnv* env) {
    delete env;
}

std::vector<double> feed(NeuropiaEnv* env, const std::vector<double>& input) {
    ASSERT(env && env->m_network.isValid());
    return env->m_network.feed(input);
}

bool setParam(NeuropiaEnv* env, const std::string& name, const std::string& value) {
   ASSERT(env);
   return env->m_params.set(name, value);
}

bool trainSimple(NeuropiaEnv* env) {
    ASSERT(env);
    Trainer trainer(env->m_root, env->m_params, true);
    trainer.train();
    env->m_network = std::move(trainer.network());
    return env->m_network.isValid();
}

bool trainEvo(NeuropiaEnv* env) {
    ASSERT(env);
    TrainerEvo trainer(env->m_root, env->m_params, true);
    trainer.train();
    env->m_network = std::move(trainer.network());
    return env->m_network.isValid();
}

bool trainParallel(NeuropiaEnv* env) {
   ASSERT(env);
   TrainerParallel trainer(env->m_root, env->m_params, true);
   trainer.train();
   env->m_network = std::move(trainer.network());
   return env->m_network.isValid();
   }

void save(NeuropiaEnv* env, const std::string& filename) {
    ASSERT(env && env->m_network.isValid());
    Neuropia::save(filename, env->m_network);
}

bool load(NeuropiaEnv* env, const std::string& filename) {
    ASSERT(env);
    const auto nets = Neuropia::load(filename);
    if(nets.size() > 0) {
        env->m_network = nets[0];
        return env->m_network.isValid();
    }
    return false;
}

int verify(NeuropiaEnv* env) {
    ASSERT(env && env->m_network.isValid());
    const auto t = Neuropia::verify(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]),
                             Neuropia::absPath(env->m_root, env->m_params["LabelsVerify"]));
    return std::get<0>(t);
}
