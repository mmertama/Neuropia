#include "trainerbase.h"
#include "utils.h"
#include "params.h"

/*
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include<string.h>

void tree(const std::string& root) {
    std::cerr << root;
    auto d = opendir(root.c_str());
    if(d) {
        struct dirent *dir;
        while((dir = readdir(d)) != nullptr) {
            if(dir->d_type == DT_REG){
                std::cerr <<  dir->d_name << std::endl;
            }
            else if (dir->d_type == DT_DIR && dir->d_name[0] != '.') {
                try {
                    tree(root + dir->d_name + "/");
                } catch (...) {
                    std::cerr << "cannot access" << dir->d_name << std::endl;
                }
            }
        }
        closedir(d);
    }
}
*/
using namespace Neuropia;

static
auto toIntVec(const std::string& s) {
    return Neuropia::Params::toVector<int>(s, [](const auto& p) noexcept {return std::atoi(p.c_str());});
}

static
auto toRealVec(const std::string& s) {
    return Neuropia::Params::toVector<NeuronType>(s, [](const auto& p) noexcept {return std::atof(p.c_str());});
}

static bool icomp(std::string_view name, std::string_view n, size_t len) {
    if(std::min(len, name.length()) != std::min(len, n.length()))
        return false;
    for(size_t i = 0; i < std::min(len, n.length()); ++i) {
        constexpr auto off = 'a' - 'A';
        static_assert(off > 0);
        if(name[i] != n[i] &&
            ((name[i] > 'z' || name[i] < 'A') || 
            !((name[i] + off ==  n[i]) || (n[i] <= 'Z' && n[i] + off == name[i]))
            ))
            return false;
        }
    return true;
}    

static
auto toFunction(const std::string& names) {
    const auto nameVec = Neuropia::Params::split(names);
    std::vector<Neuropia::ActivationFunction> functions;
    for(const auto& name : nameVec) {
        const auto cmp = [&name](std::string_view n) {return icomp(name, n, name.length());};
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
            neuropia_assert_always(false, "Invalid activation function: " + name );
    }
    return functions;
}

static
auto toInitStrategy(const std::string& strategy, const Neuropia::ActivationFunction& af) {
    if(strategy == "logistic")
        return Neuropia::Layer::InitStrategy::Logistic;
    if(strategy == "norm")
        return Neuropia::Layer::InitStrategy::Norm;
    if(strategy == "relu")
        return Neuropia::Layer::InitStrategy::ReLu;
    return Neuropia::initStrategyMap(af);
}

static
auto lrMin(const Neuropia::Params & p) {
    return p["LearningRate"] == "0" ? std::stod(p["LearningRateMin"]) : std::stod(p["LearningRate"]);
}

static
auto lrMax(const Neuropia::Params & p) {
    return p["LearningRate"] == "0" ? std::stod(p["LearningRateMax"]) : std::stod(p["LearningRate"]);
}


TrainerBase::TrainerBase(const std::string & root, const Neuropia::Params& params, bool quiet) :
    m_imageFile(Neuropia::absPath(root, params["Images"])),
    m_labelFile(Neuropia::absPath(root, params["Labels"])),
    m_images(m_imageFile),
    m_labels(m_labelFile),
    m_network(Neuropia::Layer(m_images.size(1) * m_images.size(2), toFunction(params["ActivationFunction"])[0])),
    m_dropoutRate(toRealVec(params["DropoutRate"])),
    m_start(std::chrono::high_resolution_clock::now()),
    m_learningRate(lrMax(params)),
    m_learningRateMin(lrMin(params)),
    m_learningRateMax(lrMax(params)),
    m_iterations(params.uinteger("Iterations")),
    m_testVerifyFrequency(params.uinteger("TestFrequency")),
    m_lambdaL2(params.real("L2")),
    m_quiet(quiet),
    m_classes(params.uinteger("Classes")),
    m_topology(toIntVec(params["Topology"])),
    m_afs(toFunction(params["ActivationFunction"])),
    m_initStrategy(toInitStrategy(params["InitStrategy"], toFunction(params["ActivationFunction"])[0])),
    m_maxTrainTime(params.real("MaxTrainTime")), m_control(m_maxTrainTime >= MaxTrainTime ?
                                           static_cast<decltype (m_control)>(Neuropia::timed) :
                                           static_cast<decltype (m_control)>([this](const std::function<void ()>& f, const std::string & label) {
                                               f();
                                               std::cout << (!label.empty() ? label + " " : "") << "iterations:" << m_passedIterations << std::endl;
                                               })){}

    bool TrainerBase::init() {
        m_passedIterations = 0;
        if(!m_images.ok()) {
            std::cerr << "Cannot open images from \"" << m_imageFile << "\"" << std::endl;
            return false;
        }
        if(!m_labels.ok()) {
            std::cerr << "Cannot open labels from \"" << m_labelFile << "\"" << std::endl;
            return false;
        }
        //  tree("/");

        if(m_topology.begin() == m_topology.end()) {
            std::cerr << "Bad topology " << std::endl;
            return false; 
        }



        if(m_classes == 0) {
            std::cerr << "Bad classes " << std::endl;
            return false;
        }

        m_network
        .join(m_topology.begin(), m_topology.end())
        .join(m_classes);

    for(auto i = 1U; i < m_afs.size(); i++) {
        auto npt = m_network.get(static_cast<int>(i));
        neuropia_assert_always(npt, "Too many items in list");
        npt->setActivationFunction(m_afs[i]);
    }

    m_network.initialize(m_initStrategy);
    setDropout();
    return true;
}

bool TrainerBase::isReady() const {
    return m_images.ok() && m_labels.ok() && m_network.size() > 0 && m_passedIterations < m_iterations;
}

void TrainerBase::setDropout() {
    if(!m_dropoutRate.empty()) {
        m_network.dropout(m_dropoutRate[0], true);
        for(auto i = 1U; i < m_dropoutRate.size(); i++) {
            auto npt = m_network.get(static_cast<int>(i));
            neuropia_assert_always(npt, "Too many items in list");
            npt->dropout(m_dropoutRate[i], false);
        }
    }
}

bool TrainerBase::train() {
    if(m_images.size() == 0) {
        std::cerr << "train has not initialized" << std::endl;
        return false;
        }
    if(m_images.size(1) * m_images.size(2) == 0) {
        std::cerr << "Bad dimensions" << std::endl;
        return false;
        }    
    if(m_images.size() != m_labels.size()) {
        std::cerr << "Mismatch data" << m_images.size() << " vs. " << m_labels.size() << std::endl;
        return false;
        }          
    if(m_classes == 0) {
        std::cerr << "Invalid Classes" << std::endl;
        return false;
    }
    if(m_topology.size() < 1) {
        std::cerr << "Invalid  topology" << std::endl;
        return false;
    }
    if(m_iterations < 1) {
        std::cerr << "Invalid iterations" << std::endl;
        return false;
    }
    const auto image_sz = m_images.size(1) * m_images.size(2); 
    if(static_cast<int>(image_sz) <= m_topology.front() || 
        (m_topology.size() > 1 && std::adjacent_find(m_topology.begin(), m_topology.end(), std::less<int>()) != m_topology.end()) ||
        m_topology.back() <= static_cast<int>(m_classes))  {
            std::cerr << "Fishy topology ";
            std::cerr << image_sz << ", ";
            for(const auto& i : m_topology)
                std::cerr << i << ", ";
            std::cerr << m_classes << std::endl;
            return false;
        }      
    return doTrain();
}


