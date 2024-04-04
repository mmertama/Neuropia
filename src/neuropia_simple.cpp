
#include "neuropia_simple.h"
#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "verify.h"
#include "utils.h"
#include "default.h"
#include "logstream.h"
#include "neuropia_env.h"
#include <unordered_map>


using namespace Neuropia;
using namespace NeuropiaSimple;



NeuropiaPtr NeuropiaSimple::create(const std::string& root) {
    return std::make_shared<NeuropiaSimple::NeuropiaEnv>(root);
}


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

ParamMap NeuropiaSimple::params(const NeuropiaPtr& env) {
    ASSERT(env);
    ParamMap out;
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
    default:
        neuropia_assert_always(true, "bad");    
    }
    if(!trainer->isReady()) {
        std::cerr << "Training data is not ready" << std::endl;
        return false;
    }
    if(!trainer->busy()) {
        std::cerr << "Training failed ready: " <<  Neuropia::to_string(trainer->isReady()) << ", ok: " << Neuropia::to_string(trainer->isOk()) << std::endl;
        return false;
    }
    env->m_network = std::move(trainer->network());
   return true;
}

void NeuropiaSimple::save(const NeuropiaPtr& env, const std::string& filename, SaveType saveType) {
    ASSERT(env && env->m_network.isValid());
    const auto params = env->m_params.toMap();
    Neuropia::save(filename, env->m_network, params, saveType);
}

std::optional<Neuropia::Sizes> NeuropiaSimple::load(const NeuropiaPtr& env, const std::string& filename) {
    ASSERT(env);
    const auto loaded = Neuropia::load(Neuropia::absPath(env->m_root, filename));
    if(!loaded) {
        return std::nullopt;
    }

    env->m_network = std::move(std::get<Layer>(*loaded));
    if(env->m_network.isValid()) {
        for(const auto& p : std::get<1>(*loaded)) {
            env->m_params.set(p.first, p.second);
        }
        return env->m_network.sizes();
    }
    return std::nullopt;
}

int NeuropiaSimple::verify(const NeuropiaPtr& env, size_t count) {
    ASSERT(env && env->m_network.isValid());
    Neuropia::Verifier ver(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]),
                             Neuropia::absPath(env->m_root, env->m_params["LabelsVerify"]),
                             false, 0, count, count >= static_cast<size_t>(std::numeric_limits<int>::max())); // random if not max
    const auto t = ver.busy();                         
    return std::get<0>(t);
}

void NeuropiaSimple::setLogger(const NeuropiaPtr& env, std::function<void (const std::string&)> cb) {
    ASSERT(env);
    env->setLogger([cb](const std::string& str){
        cb(str);
    });
}


const Layer& NeuropiaSimple::network(const NeuropiaPtr& env) {
    return env->m_network;
}
