#include "trainerbase.h"
#include "utils.h"
#include "params.h"

using namespace Neuropia;

auto toIntVec(const std::string& s) {
    return Neuropia::Params::toVector<int>(s, [](const auto& s){return std::atoi(s.c_str());});
}

auto toRealVec(const std::string& s) {
    return Neuropia::Params::toVector<double>(s, [](const auto& s){return std::atof(s.c_str());});
}

auto toFunction(const std::string& names) {
    const auto nameVec = Neuropia::Params::split(names);
    std::vector<Neuropia::ActivationFunction> functions;
    for(const auto& name : nameVec) {
        const auto cmp = [name](const std::string& n) {
            const auto toLower = [](std::string n) { std::transform(n.begin(), n.end(), n.begin(), [](auto c){
                    return c >= 'A' && c <= 'Z' ? c - ('A' - 'a') : c; }); return n;};
            return toLower(n.substr(0, n.length() - std::string("Function").length())) == name;};
        if(cmp(Neuropia::signumFunction.name()))
            functions.push_back(Neuropia::signumFunction);
        else if(cmp(Neuropia::binaryFunction.name()))
            functions.push_back(Neuropia::binaryFunction);
        else if(cmp(Neuropia::sigmoidFunction.name()))
            functions.push_back(Neuropia::sigmoidFunction);
        else if(cmp(Neuropia::reLuFunction.name()))
            functions.push_back(Neuropia::reLuFunction);
        else if(cmp(Neuropia::eluFunction.name()))
            functions.push_back(Neuropia::eluFunction);
        else
            neuropia_assert_always(false, "Invalid activation function");
    }
    return functions;
}

auto toInitStrategy(const std::string& strategy, Neuropia::ActivationFunction af) {
    if(strategy == "logistic")
        return Neuropia::Layer::InitStrategy::Logistic;
    if(strategy == "norm")
        return Neuropia::Layer::InitStrategy::Norm;
    if(strategy == "relu")
        return Neuropia::Layer::InitStrategy::ReLu;
    return Neuropia::initStrategyMap(af);
}

auto lrMin(const Neuropia::Params & p) {
    return p["LearningRate"] == "0" ? std::stod(p["LearningRateMin"]) : std::stod(p["LearningRate"]);
}

auto lrMax(const Neuropia::Params & p) {
    return p["LearningRate"] == "0" ? std::stod(p["LearningRateMax"]) : std::stod(p["LearningRate"]);
}


TrainerBase::TrainerBase(const std::string & root, const Neuropia::Params& params, bool quiet) :
    imageFile(Neuropia::absPath(root, params["Images"])),
    labelFile(Neuropia::absPath(root, params["Labels"])),
    images(imageFile),
    labels(labelFile),
    m_network(Neuropia::Layer(images.size(1) * images.size(2), toFunction(params["ActivationFunction"])[0])),
    dropoutRate(toRealVec(params["DropoutRate"])),
    start(std::chrono::high_resolution_clock::now()),
    learningRate(lrMax(params)),
    learningRateMin(lrMin(params)),
    learningRateMax(lrMax(params)),
    iterations(params.uinteger("Iterations")),
    testVerifyFrequency(params.uinteger("TestFrequency")),
    lambdaL2(params.real("L2")),
    quiet(quiet),
    maxTrainTime(params.real("MaxTrainTime")),control(maxTrainTime >= MaxTrainTime ?
                                           static_cast<decltype (control)>(Neuropia::timed) :
                                           static_cast<decltype (control)>([this](const std::function<void ()>& f, const std::string & label) {
                                               f();
                                               std::cout << (label.size() > 0 ? label + " " : "") << "iterations:" << passedIterations << std::endl;
                                               })){

    neuropia_assert_always(images.ok(), imageFile);
    neuropia_assert_always(labels.ok(), labelFile);
    const auto topology = toIntVec(params["Topology"]);
    const auto afs = toFunction(params["ActivationFunction"]);
    const auto initStrategy = toInitStrategy(params["InitStrategy"], toFunction(params["ActivationFunction"])[0]);

    m_network
    .join(topology.begin(), topology.end())
    .join(params.uinteger("Classes"));

    for(auto i = 1U; i < afs.size(); i++) {
        auto npt = m_network.get(static_cast<int>(i));
        neuropia_assert_always(npt, "Too many items in list");
        npt->setActivationFunction(afs[i]);
    }

    m_network.initialize(initStrategy);
    setDropout();
}

void TrainerBase::setDropout() {
    if(dropoutRate.size() > 0) {
        m_network.dropout(dropoutRate[0], true);
        for(auto i = 1U; i < dropoutRate.size(); i++) {
            auto npt = m_network.get(static_cast<int>(i));
            neuropia_assert_always(npt, "Too many items in list");
            npt->dropout(dropoutRate[i], false);
        }
    }
}



