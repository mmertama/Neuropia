#ifndef TRAINCOMMON_H
#define TRAINCOMMON_H

#include <iostream>
#include <vector>
#include "neuropia.h"
#include "idxreader.h"
#include "utils.h"


namespace Neuropia {
    class Params;


constexpr NeuronType MaxTrainTime = 999999;

class TrainerBase {
public:
    TrainerBase(const std::string& root, const Neuropia::Params& params, bool quiet);
    virtual ~TrainerBase() = default;
    Neuropia::Layer&& network() {return std::move(m_network);}
    bool isReady() const;
    bool next();
    bool busy();
    bool isOk() const;
    virtual bool complete() = 0; 
protected:
    virtual bool doTrain() = 0;
    virtual bool doInit() = 0;
    bool verify();
    unsigned iterations() const {return m_iterations;}
    unsigned classes() const {return m_classes;}
    bool update();
private:
    bool init();
    void setDropout();
protected:
    const std::string m_imageFile;
    const std::string m_labelFile;
    Neuropia::IdxReader<unsigned char> m_images;
    Neuropia::IdxReader<unsigned char> m_labels;
    Neuropia::Layer m_network;
    Neuropia::Random m_random = {};
    NeuronType m_learningRate;
    const NeuronType m_lambdaL2;
private:
    const std::vector<NeuronType> m_dropoutRate;
    std::chrono::high_resolution_clock::time_point m_start;
    const NeuronType m_learningRateMin;
    const NeuronType m_learningRateMax;
    NeuronType m_gap = 0;
    const unsigned m_testVerifyFrequency;
    const unsigned m_testSamples;
    const bool m_quiet;
    const unsigned m_classes;
    const std::vector<int> m_topology;
    const std::vector<Neuropia::ActivationFunction> m_afs;
    const Layer::InitStrategy m_initStrategy;
    const NeuronType m_maxTrainTime;
    const std::function<void (const std::function<void ()>&, const std::string&)> m_control;
private:
    const unsigned m_iterations;
    unsigned m_current = 0;
    unsigned m_testVerify = 0;
};
}






#endif // TRAINCOMMON_H
