
#include "neuropia_simple.h"
#include <unordered_map>
#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "verify.h"
#include "default.h"
#include "simple.h"

using namespace Neuropia;
using namespace NeuropiaSimple;

bool fatal(const char* t, const char* f, int line, const char* file) {
    std::cerr << "Assert:" << t << " in line " << line << " at " << f << "  in "<< file << "." << std::endl;
    std::abort();
}

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
    if(!trainer->init()) {
        std::cerr << "Invalid trainer" << std::endl;
        return false;
    }
    if(!trainer->isReady()) {
        std::cerr << "Training data is not ready" << std::endl;
        return false;
    }
    trainer->train();
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

int NeuropiaSimple::verify(const NeuropiaPtr& env) {
    ASSERT(env && env->m_network.isValid());
    const auto t = Neuropia::verify(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]),
                             Neuropia::absPath(env->m_root, env->m_params["LabelsVerify"]),
                             false);
    return std::get<0>(t);
}

void NeuropiaSimple::setLogger(const NeuropiaPtr& env, std::function<void (const std::string&)> cb) {
    ASSERT(env);
    env->setLogger([cb](const std::string& str){
        cb(str);
    });
    std::cout << "Neuropia loaded" << std::endl;
}


const Layer& NeuropiaSimple::network(const NeuropiaPtr& env) {
    return env->m_network;
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
            percentage(m_passedIterations, m_iterations);
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
            percentage(delta, m_maxTrainTime, " " + std::to_string(m_learningRate));
        this->m_learningRate += (static_cast<NeuronType>(change) / static_cast<NeuronType>(m_maxTrainTime)) * (m_learningRateMin - m_learningRateMax);
    }

    const auto imageSize = m_images.size(1) * m_images.size(2);
    const auto at = m_random.random(m_images.size());
    const auto image = m_images.readAt(at, imageSize);
    const auto label = static_cast<unsigned>(m_labels.readAt(at));

#ifdef DEBUG_SHOW
    Neuropia::printimage(image.data(), m_images.size(1), m_images.size(2)); //ASCII print images
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
        Neuropia::percentage(m_position, sz);
    }

    return true;
}


