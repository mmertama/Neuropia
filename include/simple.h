#pragma once

#include "params.h"
#include "evotrain.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "default.h"


bool fatal(const char* t, const char* f, int line, const char* file);

#define ASSERT(X) ((X) || fatal("Invalid", __FUNCTION__, __LINE__, __FILE__))
#define ASSERT_X(X, T) ((X) || fatal((T), __FUNCTION__, __LINE__, __FILE__))




namespace NeuropiaSimple {

template <size_t SZ>
class LogStream : public std::streambuf {
public:
    LogStream(std::function<void (const std::string&)>& logger) : m_logger(logger), m_os(this) {
        setp(m_buffer, m_buffer + SZ - 1);
    }

    ~LogStream() override {
    }
    /**
     * Since m_logger can be very very slow (on WASM) the frequent updates can be frozen
     */
    void freeze(bool doFreeze) {
        m_freeze = doFreeze;
    }

private:
    int_type overflow(int_type ch) override {
        if(ch != traits_type::eof()){
            *pptr() = static_cast<char>(ch);
            pbump(1);
            write();
        }
        return ch;
    }
    int sync() override {
        write();
        return 1;
    }
    void write() {
        const auto n = static_cast<size_t>(pptr() - pbase());
        if(!m_freeze) {
            const auto buf = std::string(m_buffer, n);
            m_logger(buf);
        }
        pbump(-(static_cast<int>(n)));
    }
private:
    std::function<void (const std::string&)> m_logger;
    char m_buffer[SZ];
    std::ostream m_os;
    bool m_freeze = false;
};

using SimpleLogStream = LogStream<2048>;

class SimpleTrainer : public Neuropia::TrainerBase {
public:
    SimpleTrainer(const std::string& root, const Neuropia::Params& params, bool quiet, const std::function<void (Neuropia::Layer&& layer, bool)>& onEnd) : 
        Neuropia::TrainerBase(root, params, quiet), m_onEnd(onEnd) {}
    bool train(unsigned its, SimpleLogStream& stream);
    bool isOk() const {return isReady() && m_ok;}

private:
    bool train();
    std::function<void (Neuropia::Layer&& layer, bool)> m_onEnd;
    bool m_ok = true;
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

    std::unique_ptr<SimpleTrainer> m_trainer = {};
    std::unique_ptr<SimpleVerifier> m_verifier = {};
};

}
