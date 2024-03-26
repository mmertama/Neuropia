

#include "neuropia_simple.h"
#include <unordered_map>
#include "simple.h"
#include "verify.h"
#include "default.h"

#include <emscripten/bind.h>

using namespace emscripten;

static
bool train(const NeuropiaSimple::NeuropiaPtr& env, unsigned iteration) {
    ASSERT(env);

    if(iteration == 0 || !env->m_trainer) {
        env->m_logStream->freeze(false);
        std::cout << "Create trainer"  << std::endl;;
        env->m_trainer = std::make_unique<NeuropiaSimple::SimpleTrainer>(env->m_root, env->m_params, false, [env](Neuropia::Layer&& layer, bool){
            env->m_network = layer;
        });
    }

    return env->m_trainer->train(iteration, *env->m_logStream);
}

static
bool verify(const NeuropiaSimple::NeuropiaPtr& env, int iteration) {
    if(iteration == 0 || !env->m_verifier) {
        ASSERT(env->m_network.isValid());
        env->m_logStream->freeze(false);
        std::cout << "Create verifier"  << std::endl;;
        env->m_verifier.reset(new NeuropiaSimple::SimpleVerifier(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]), Neuropia::absPath(env->m_root,env->m_params["LabelsVerify"])));
        Neuropia::percentage(0, 1);
    }
    return env->m_verifier->verify();
}


static
NeuropiaSimple::ParamMap basicParams(const NeuropiaSimple::NeuropiaPtr& env) {
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
bool isNetworkValid(const NeuropiaSimple::NeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_trainer) {
        env->m_logStream->freeze(false);
    }
    return env->m_network.isValid();
}

static
Neuropia::NeuronType verifyResult(const NeuropiaSimple::NeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_verifier) {
        env->m_logStream->freeze(false);
        return env->m_verifier->verifyResult();
    }
    return -2.0; //arbitrary negative number, but not -1
}


static
int showImage(const NeuropiaSimple::NeuropiaPtr& env, const std::string& imageName, const std::string& labelName, unsigned index) {
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


void setLogger(const NeuropiaSimple::NeuropiaPtr& env, emscripten::val cb) {
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

std::vector<Neuropia::NeuronType> feed(NeuropiaSimple::NeuropiaPtr env, emscripten::val a) {
    const auto image = fromJSArray<Neuropia::NeuronType>(a);

    std::vector<unsigned char> test(image.size());
    std::transform(image.begin(), image.end(), test.begin(), [](Neuropia::NeuronType c) {
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
    std::transform(image.begin(), image.end(), inputs.begin(), [](Neuropia::NeuronType c) {
        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0., 255.);
    });
    return NeuropiaSimple::feed(env, inputs);
}

// just help find type (cast would do as well)
static 
bool setParam(const NeuropiaSimple::NeuropiaPtr& env, const std::string& name, const std::string& value) {
   return env->m_params.set(name, value);
}

static 
bool load(const NeuropiaSimple::NeuropiaPtr& env, const std::string& filename) {
    return NeuropiaSimple::load(env, filename).has_value();
}

static 
bool create() {
    return NeuropiaSimple::create();
}


EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<NeuropiaSimple::NeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<NeuropiaSimple::NeuropiaEnv, const std::string&>);
    register_vector<Neuropia::NeuronType>("ValueVector");
    register_vector<std::string>("StringVector");
    register_map<NeuropiaSimple::ParamMap::key_type, NeuropiaSimple::ParamMap::mapped_type>("ParamMap");
    function("create", &::create);
    function("free", &NeuropiaSimple::free);
    function("feed", &::feed);
    function("setParam", &::setParam);
    function("params", &::basicParams);
    function("train", &::train);
    function("save", &NeuropiaSimple::save);
    function("load", &::load);
    function("verify", &::verify);
    function("setLogger", &::setLogger);
    function("isNetworkValid", &::isNetworkValid);
    function("verifyResult", &::verifyResult);
    function("showImage", &::showImage);
}
