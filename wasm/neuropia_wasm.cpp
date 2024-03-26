

#include "neuropia_simple.h"
#include <unordered_map>
#include "simple.h"
#include "verify.h"
#include "default.h"

using namespace NeuropiaSimple;
using namespace Neuropia;

static
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

static
bool verify(const NeuropiaPtr& env, int iteration) {
    if(iteration == 0 || !env->m_verifier) {
        ASSERT(env->m_network.isValid());
        env->m_logStream->freeze(false);
        std::cout << "Create verifier"  << std::endl;;
        env->m_verifier.reset(new SimpleVerifier(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]), Neuropia::absPath(env->m_root,env->m_params["LabelsVerify"])));
        Neuropia::percentage(0, 1);
    }
    return env->m_verifier->verify();
}


static
ParamMap basicParams(const NeuropiaPtr& env) {
    auto p = NeuropiaSimple::params(env);
    p.erase("Jobs");    //mt
    p.erase("BatchSize"); //mt
    p.erase("BatchVerifySize"); //mt
    p.erase("Hard"); //for ensenble
    p.erase("Extra"); //this is for ensemble
    p.erase("LearningRate"); //use only min and max
    return p;
}

static
bool isNetworkValid(const NeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_trainer) {
        env->m_logStream->freeze(false);
    }
    return env->m_network.isValid();
}

static
NeuronType verifyResult(const NeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_verifier) {
        env->m_logStream->freeze(false);
        return env->m_verifier->verifyResult();
    }
    return -2.0; //arbitrary negative number, but not -1
}


static
int showImage(const NeuropiaPtr& env, const std::string& imageName, const std::string& labelName, unsigned index) {
    ASSERT(env);
    Neuropia::IdxReader<std::array<unsigned char, 28 * 28>> idxi(Neuropia::absPath(env->m_root, imageName));
    if(!idxi.ok() || idxi.dimensions() != 3)
        return -1;

    Neuropia::IdxReader<unsigned char> idxl(Neuropia::absPath(env->m_root, labelName));
    if(!idxl.ok() || idxl.dimensions() != 1)
        return -2;

    const auto data = idxi.readAt(index);
    Neuropia::printimage(data.data(), 28, 28);
    return idxl.readAt(index);
}


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
    Neuropia::printimage(test.data(), 28, 28);
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







EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<NeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<NeuropiaEnv, const std::string&>);
    register_vector<NeuronType>("ValueVector");
    register_vector<std::string>("StringVector");
    register_map<ParamMap::key_type, ParamMap::mapped_type>("ParamMap");
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
