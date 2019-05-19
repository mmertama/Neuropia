#include "idxreader.h"
#include <fstream>
#include <string>
#include <cassert>

//http://yann.lecun.com/exdb/mnist/

using namespace Neuropia;

#ifndef LITTLE_ENDIAN
#if 'ABCD' == 0x41424344
#define LITTLE_ENDIAN
#else
#ifndef BIG_ENDIAN
#define BIG_ENDIAN
#endif
#endif
#endif

size_t IdxReaderBase::read(char* data, size_t size, size_t n) {
   size_t count = 0;
#if defined(LITTLE_ENDIAN) || !defined(BIG_ENDIAN)
    assert(size > 0  && n > 0);
    if(size == 1) {
        m_stream.read(data, static_cast<std::streamsize>(size * n));
        count = static_cast<size_t>(m_stream.gcount());
    } else {
        for(auto d = 0ULL; d < size * n; d += size) {
            readBE(data + d, size);
            count += static_cast<size_t>(m_stream.gcount());
        }
    }
#else
    m_stream.read(data, size * n);
    count = m_stream.gcount();
#endif
    if(count < size * n) {
        m_type = Type::Invalid;
    }
    return static_cast<size_t>(count);
}

void IdxReaderBase::readBE(char* data, size_t size) {
#if defined(LITTLE_ENDIAN) || !defined(BIG_ENDIAN)
    switch(size) {
    case 1: {
        m_stream.read(data, 1);
    }
    break;
    case 2: {
        char buffer[2];
        m_stream.read(static_cast<char*>(buffer), 2);
        data[0] = buffer[1];
        data[1] = buffer[0];
    }
    break;
    case 4: {
        char buffer[4];
        m_stream.read(static_cast<char*>(buffer), 4);
        data[0] = buffer[3];
        data[1] = buffer[2];
        data[2] = buffer[1];
        data[3] = buffer[0];
    }
    break;
    case 8: {
        char buffer[8];
        m_stream.read(static_cast<char*>(buffer), 8);
        data[0] = buffer[3];
        data[1] = buffer[2];
        data[2] = buffer[1];
        data[3] = buffer[0];
        data[4] = buffer[4];
        data[5] = buffer[5];
        data[6] = buffer[6];
        data[7] = buffer[7];
    }
    break;
    default:
        assert(false);
    }
#endif
}

#define BSZ(x) ((x) * 1024)

IdxReaderBase::IdxReaderBase(const std::string& name, unsigned iobufsz) : m_iobuf(iobufsz > 0 ? new char[BSZ(iobufsz)] : nullptr) {
    if(iobufsz > 0) {
        m_stream.rdbuf()->pubsetbuf(m_iobuf.get(), BSZ(iobufsz));
    }
    m_stream.open(name, std::ios::binary);
    if(m_stream.is_open()) {
        unsigned magic;
        read(reinterpret_cast<char*>(&magic), 4, 1);
        if((magic >> 16) == 0) {
            constexpr Type T[] = {Type::Invalid, Type::Invalid, Type::Invalid, Type::Invalid,
                                  Type::Invalid, Type::Invalid, Type::Invalid, Type::Invalid,
                                  Type::Byte, Type::Char, Type::Invalid, Type::Short,
                                  Type::Int, Type::Float, Type::Double, Type::Invalid
                                 };
            m_type = T[magic >> 8];
            const auto dimensions = magic & 0xF;
            for(size_t i = 0; i < dimensions; i++) {
                unsigned dim;
                if(4 != read(reinterpret_cast<char*>(&dim), 4, 1)) {
                    m_type = Type::Invalid;
                    return;
                }
                m_dimensions.push_back(dim);
            }
        }
        m_headerSize = position();
    }
}

void IdxReaderBase::moveTo(size_t position) {
    m_stream.seekg(static_cast<std::streamoff>(m_headerSize + position));
}
