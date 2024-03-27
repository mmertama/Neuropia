#pragma once

#include <functional>
#include <ostream>

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

}