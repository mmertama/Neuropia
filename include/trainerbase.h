#ifndef TRAINCOMMON_H
#define TRAINCOMMON_H

#include <iostream>
#include <vector>
#include "neuropia.h"
#include "idxreader.h"

constexpr double MaxTrainTime = 999999;

namespace Neuropia {
    class Params;

class TrainerBase {
public:
    enum class Show{Nothing, Progress};
    TrainerBase(const std::string & root, const Neuropia::Params& params, bool quiet);
    void setDropout();
    virtual bool train() = 0;
    virtual ~TrainerBase() = default;
    Neuropia::Layer&& network() {return std::move(m_network);}
public:
    const std::string imageFile;
    const std::string labelFile;
    Neuropia::IdxRandomReader<unsigned char> images;
    Neuropia::IdxRandomReader<unsigned char> labels;
    Neuropia::Layer m_network;
    const std::vector<double> dropoutRate;
    size_t passedIterations = 0;
    std::chrono::high_resolution_clock::time_point start;
    double learningRate;
    const double learningRateMin;
    const double learningRateMax;
    double gap = 0;
    unsigned iterations;
    unsigned testVerifyFrequency;
    double lambdaL2;
    bool quiet;
    const double maxTrainTime;
    const std::function<void (const std::function<void ()>&, const std::string&)> control;
};
}






#endif // TRAINCOMMON_H
