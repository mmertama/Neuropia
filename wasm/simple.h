#pragma once

#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "default.h"
#include "logstream.h"


namespace NeuropiaSimple {


class SimpleTrainer : public Neuropia::TrainerBase {
public:
    SimpleTrainer(const std::string& root, const Neuropia::Params& params, bool quiet, const std::function<void (Neuropia::Layer&& layer, bool)>& onEnd) : 
        Neuropia::TrainerBase(root, params, quiet), m_onEnd(onEnd) {}
    bool train(unsigned batchSize, SimpleLogStream& stream);

private:
    bool doTrain() override;
    std::function<void (Neuropia::Layer&& layer, bool)> m_onEnd;

};


constexpr unsigned IoBufSz = 5;

class SimpleVerifier {
public:
    SimpleVerifier(const Neuropia::Layer& network, const std::string& imageFile, const std::string& labelFile) : m_network(network),
        m_testImages(imageFile, IoBufSz), m_testLabels(labelFile, IoBufSz) {
    }
    bool verify();
    Neuropia::NeuronType verifyResult() const {
        return m_testLabels.size() ? static_cast<Neuropia::NeuronType>(m_found) / static_cast<Neuropia::NeuronType>(m_testLabels.size()) : -1.0;
        }
private:
    const Neuropia::Layer& m_network;
    Neuropia::IdxReader<unsigned char> m_testImages;
    Neuropia::IdxReader<unsigned char> m_testLabels;
    size_t m_position = 0;
    unsigned m_found = 0;
};


}
