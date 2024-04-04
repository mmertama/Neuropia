#pragma once

#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "verify.h"
#include "default.h"
#include "logstream.h"


/**
 * TODO Remove these Simple implementations, they are just non-blocking  reimplementations of Trainer and Verifier - refactor
 * those they can be used in non-blocking envs (like WASM)
 */
namespace NeuropiaSimple {

class SimpleTrainer {
public:
    SimpleTrainer(const std::string& root, const Neuropia::Params& params, bool quiet, const std::function<void (Neuropia::Layer&& layer, bool)>& onEnd) : 
        m_trainer(root, params, quiet), m_onEnd(onEnd) {}
    
    bool train() {
        const bool ok = m_trainer.next();
        if(!ok)
            m_onEnd(m_trainer.network(), m_trainer.isOk());
        return ok ||  m_trainer.isOk();
    }

    bool isReady() const {return m_trainer.isReady();}
private:
    Neuropia::Trainer m_trainer;
    std::function<void (Neuropia::Layer&& layer, bool)> m_onEnd;

};


constexpr unsigned IoBufSz = 5;

class SimpleVerifier {
public:
    SimpleVerifier(const Neuropia::Layer& network, const std::string& imageFile, const std::string& labelFile) : m_verifier(network, imageFile, labelFile, IoBufSz) {
    }
    
    bool verify() {
        const bool ok = m_verifier.next();
    return ok;
    }

    Neuropia::NeuronType verifyResult() const {
        return std::get<1>(m_verifier.result());
    }
private:
    Neuropia::Verifier m_verifier;
};





}
