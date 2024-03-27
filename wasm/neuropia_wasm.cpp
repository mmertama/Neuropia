

#include "neuropia_simple.h"
#include <unordered_map>
#include "simple.h"
#include "verify.h"
#include "default.h"
#include "neuropia_env.h"
#include <emscripten/bind.h>

using namespace emscripten;

class WasmNeuropiaEnv : public NeuropiaSimple::NeuropiaEnv {
    public:
        explicit WasmNeuropiaEnv(const std::string& root) : NeuropiaSimple::NeuropiaEnv(root) {}
        std::unique_ptr<NeuropiaSimple::SimpleTrainer> m_trainer;
        std::unique_ptr<NeuropiaSimple::SimpleVerifier> m_verifier;
};

using WasmNeuropiaPtr = std::shared_ptr<WasmNeuropiaEnv>;

static
bool train(const WasmNeuropiaPtr& env, unsigned batch_size) {
    ASSERT(env);

    if(!env->m_trainer || !env->m_trainer->isReady()) {
        env->m_logStream->freeze(false);
        std::cout << "Start training" << std::endl;
        env->m_trainer = std::make_unique<NeuropiaSimple::SimpleTrainer>(env->m_root, env->m_params, false, [env](Neuropia::Layer&& layer, bool) {
            env->m_network = layer;
        });
         
        if(!env->m_trainer->init()) {
            std::cerr << "Init failed" << std::endl;
            return false;
        }
    }
    return env->m_trainer->train(batch_size, *env->m_logStream);
}

static
bool verify(const WasmNeuropiaPtr& env, int iteration) {
    if(iteration == 0 || !env->m_verifier) {
        ASSERT(env->m_network.isValid());
        env->m_logStream->freeze(false);
        std::cout << "Create verifier" << std::endl;;
        env->m_verifier = std::make_unique<NeuropiaSimple::SimpleVerifier>(env->m_network, Neuropia::absPath(env->m_root, env->m_params["ImagesVerify"]), Neuropia::absPath(env->m_root,env->m_params["LabelsVerify"]));
        Neuropia::percentage(0, 1);
    }
    return env->m_verifier->verify();
}


static
NeuropiaSimple::ParamMap basicParams(const WasmNeuropiaPtr& env) {
    auto p = NeuropiaSimple::params(env);
    p.erase("Jobs");    //mt
    p.erase("BatchSize"); //mt
    p.erase("BatchVerifySize"); //mt
    p.erase("Hard"); //for ensenble
    p.erase("Extra"); //this is for ensemble
    p.erase("LearningRate"); //use only min and max
    // tweak the UI
    //const auto iterations = std::stoi(p["Iterations"]);
    //const auto ui_iterations = iterations * ITERATION_BATCH_SIZE;
    //p["Iterations"] = std::to_string(ui_iterations);
    return p;
}

static
bool isNetworkValid(const WasmNeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_trainer) {
        env->m_logStream->freeze(false);
    }
    return env->m_network.isValid();
}

static
Neuropia::NeuronType verifyResult(const WasmNeuropiaPtr& env) {
    ASSERT(env);
    if(env->m_verifier) {
        env->m_logStream->freeze(false);
        return env->m_verifier->verifyResult();
    }
    return -2.0; //arbitrary negative number, but not -1
}


static
int showImage(const WasmNeuropiaPtr& env, const std::string& imageName, const std::string& labelName, unsigned index) {
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


void setLogger(const WasmNeuropiaPtr& env, emscripten::val cb) {
    NeuropiaSimple::setLogger(env, [cb](const std::string& str){
        cb(str);
    });
    std::cout << "Neuropia logger installed" << std::endl;
}


template <typename T>
std::vector<T> fromJSArray(const emscripten::val& v) {
    const auto l = v["length"].as<unsigned>();
    std::vector<T> vec(l);
    emscripten::val memoryView(emscripten::typed_memory_view(l, vec.data()));
    memoryView.call<void>("set", v);
    return vec;
}

std::vector<Neuropia::NeuronType> feed(WasmNeuropiaPtr env, emscripten::val a) {
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
bool setParam(const WasmNeuropiaPtr& env, const std::string& name, const std::string& value) {
    //if(name == "Iterations") { // tweaked ui
    //    const auto ui_iterations = std::stoi(value);
    //    const auto iterations = iterations / ITERATION_BATCH_SIZE;
    //    return env->m_params.set(name, std::to_string(iterations));
    //} else
    return env->m_params.set(name, value);
}

static 
bool load(const WasmNeuropiaPtr& env, const std::string& filename) {
    return NeuropiaSimple::load(env, filename).has_value();
}

static 
int batchSize() {
    return ITERATION_BATCH_SIZE;
}

static
auto create(const std::string root) {
     return std::make_shared<WasmNeuropiaEnv>(root);
}


EMSCRIPTEN_BINDINGS(Neuropia) {
    class_<WasmNeuropiaEnv>("Neuropia").smart_ptr_constructor("Neuropia", &std::make_shared<WasmNeuropiaEnv, const std::string&>);
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
    function("batchSize", &::batchSize);
}
