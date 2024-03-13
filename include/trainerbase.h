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
    enum class Show{Nothing, Progress};
    TrainerBase(const std::string& root, const Neuropia::Params& params, bool quiet);
    void setDropout();
    virtual bool train() = 0;
    virtual ~TrainerBase() = default;
    Neuropia::Layer&& network() {return std::move(m_network);}
    bool isReady() const;
protected:
    const std::string m_imageFile;
    const std::string m_labelFile;
    Neuropia::IdxReader<unsigned char> m_images;
    Neuropia::IdxReader<unsigned char> m_labels;
    Neuropia::Layer m_network;
    const std::vector<NeuronType> m_dropoutRate;
    size_t m_passedIterations = 0;
    std::chrono::high_resolution_clock::time_point m_start;
    NeuronType m_learningRate;
    const NeuronType m_learningRateMin;
    const NeuronType m_learningRateMax;
    NeuronType m_gap = 0;
    unsigned m_iterations;
    unsigned m_testVerifyFrequency;
    NeuronType m_lambdaL2;
    bool m_quiet;
    const NeuronType m_maxTrainTime;
    const std::function<void (const std::function<void ()>&, const std::string&)> m_control;
    Neuropia::Random m_random;
};
}






#endif // TRAINCOMMON_H
