#pragma once
#include "default.h"
#include "neuropia.h"
#include "params.h"
#include "logstream.h"
#include "trainerbase.h"
#include <iostream>

namespace NeuropiaSimple { 

class NeuropiaEnv {
public:
    NeuropiaEnv(const NeuropiaEnv&) = delete;
    NeuropiaEnv& operator=(const NeuropiaEnv&) = delete;
    NeuropiaEnv(const std::string& root) :  m_root(root) {
        m_params.addHelp( topologyRe, "\',\'-separated list of integers");
        m_params.addHelp( activationFunctionRe, "\',\'-separated list of activation functions: \"sigmoid, relu or elu\"");
        m_params.addHelp( dropoutRateRe, "\',\'-separated of list of real numbers");
    }
    virtual ~NeuropiaEnv();
    void setLogger(std::function<void (const std::string&)> logger) {
        if(m_prevStreamBufCerr) {
            std::cerr.rdbuf(m_prevStreamBufCerr);
        }
        if(m_prevStreamBufCout) {
            std::cout.rdbuf(m_prevStreamBufCout);
        }
        m_logStream = std::make_unique<NeuropiaSimple::SimpleLogStream>(logger);
        m_prevStreamBufCout = std::cout.rdbuf(m_logStream.get());
        m_prevStreamBufCerr = std::cerr.rdbuf(m_logStream.get());
    }
    Neuropia::Layer m_network = {};
    Neuropia::Params m_params = {
        DEFAULT_PARAMS
};
    const std::string m_root;
    std::unique_ptr<SimpleLogStream> m_logStream = {};
    std::streambuf* m_prevStreamBufCout = nullptr;
    std::streambuf* m_prevStreamBufCerr = nullptr;
    bool m_once = false;
};

}

