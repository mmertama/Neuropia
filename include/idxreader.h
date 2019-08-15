#ifndef IDXREADER_H
#define IDXREADER_H
#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <random>
#include <chrono>



namespace Neuropia {

constexpr unsigned DefaultIobufSzRandom = 2;
constexpr unsigned DefaultIobufSz = 200;

/**
 * @brief The IdxReade class reads IDX files
 */

/**
 * @brief Dont create a IdxReaderBase, create IdxReader of your data type instead
 */
class IdxReaderBase {
public:
    enum class Type{Invalid, Byte, Char, Short, Int, Float, Double};
    IdxReaderBase(const std::string& name, unsigned iobufszKB);
    /**
     * @brief type
     * @return type of data get from file
     */

    Type type() const {return m_type;}
    /**
     * @brief elementSize
     * @return size of that type
     */
    size_t elementSize() const {constexpr size_t P[] = {0, 1, 1, 2, 4, 4, 8}; return P[static_cast<int>(type())];}

    /**
     * @brief dimensions
     * @return number of data dimensions, 1 for array of single values, 2 array of arrays, 3, array of matrices etc....
     */
    size_t dimensions() const {return m_dimensions.size();}

    /**
     * @brief dimension
     * @param dim
     * @return size, typically 0 number of data elements
     */
    size_t size(size_t dim = 0) const {return dim < m_dimensions.size() ? m_dimensions[dim] : 0;}

    /**
     * @brief position
     * @return position at readong
     */
    size_t position() const {return static_cast<size_t>(m_stream.tellg());}

    /**
     * @brief ok
     * @return file is open and dandy
     */
    bool ok() const {return m_stream.good() && size() > 0 && m_type != Type::Invalid;}

protected:
    /**
     * @brief read inherit your own class if dynamic sizes are needed and use this to access bytes
     * @param b buffer where data is read to
     * @param size byte size of element
     * @param n amount of the elements to read, note that in little endian device read(b, 1, 64) is different than read(b, 8, 8) even amount of read bytes is the same, later
     * may have bytes swapped to native as IDX data is BE.
     */
    size_t read(char* data, size_t size, size_t n);
    void moveTo(size_t dataPosition);
private:
    void readBE(char* data, size_t size);
private:
    mutable std::ifstream m_stream;
    Type m_type = Type::Invalid;
    std::vector<size_t> m_dimensions;
    std::unique_ptr<char[]> m_iobuf;
    size_t m_headerSize;
};

template <typename T>
/**
 * @brief The IdxReader class single element data, e.g. Bytes, Ints, Doubles...
 */
class IdxReader : public IdxReaderBase {
public:
    /**
     * @brief IdxReader
     * @param name filename
     * @param iobufsz buffer size, default is DefaultIobufSz
     */
    IdxReader(const std::string& name, unsigned iobufsz = DefaultIobufSz) : IdxReaderBase(name, iobufsz) {}

    /**
     * @brief next
     * @return iterate over data, see dimension(0) how many times to be done
     */
    T next() {
        T data;
        read(reinterpret_cast<char*>(&data), sizeof(T), 1);
        return data;
    }

    /**
     * @brief next
     * @param sz
     * @return iterate over data, see dimension(0) how many times to be done
     */
    std::vector<T> next(size_t sz) {
        std::vector<T> data(sz);
        read(reinterpret_cast<char*>(data.data()), sizeof(T), sz);
        return data;
    }
};


template <typename T, size_t S>
/**
 * @brief The IdxReader<std::array<T, S> > class for data arrays
 */
class IdxReader<std::array<T, S>> : public IdxReaderBase {
public:
    /**
     * @brief IdxReader
     * @param name as above
     * @param iobufsz as above
     */
    IdxReader(const std::string& name, unsigned iobufsz = DefaultIobufSz) : IdxReaderBase(name, iobufsz) {}
    /**
     * @brief next
     * @return array of data.  see dimension(0) how many times to be done -  then other dimensions will defines what would be the size.
     * Except it is expected that user know data dimensions before hand as array size is give compile time.
     */
    std::array<T, S> next() {
        std::array<T, S> data;
        read(reinterpret_cast<char*>(data.data()),  sizeof(T), S);
        return data;
    }
};


class IdxRandomReaderBase : public IdxReaderBase {
public:
    IdxRandomReaderBase(const std::string& name, unsigned seed, unsigned iobufsz = DefaultIobufSz)
        : IdxReaderBase(name, iobufsz),  m_gen(seed) {}
    /**
     * @brief random
     * @return
     */
    size_t random() const {
        return (m_gen() % size());
    }
private:
    mutable std::default_random_engine m_gen;
};

template <typename T>
/**
 * @brief The IdxReader class single element data, e.g. Bytes, Ints, Doubles...
 */
class IdxRandomReader : public IdxRandomReaderBase {
public:
    /**
     * @brief IdxReader
     * @param name filename
     * @param iobufsz buffer size, default is DefaultIobufSz
     */
    IdxRandomReader(const std::string& name,
                    unsigned seed =
        #ifndef RANDOM_SEED
            static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
        #else
            RANDOM_SEED
        #endif
                    , unsigned iobufsz = DefaultIobufSzRandom) : IdxRandomReaderBase(name, seed, iobufsz) {}

    /**
     * @brief next
     * @param pos
     * @return iterate over data, see dimension(0) how many times to be done
     */
    T next(size_t pos) {
        T data;
        moveTo(pos * sizeof(T));
        read(reinterpret_cast<char*>(&data), sizeof(T), 1);
        return data;
    }

    /**
     * @brief next
     * @param pos
     * @param sz
     * @return
     */
    std::vector<T> next(size_t pos, size_t sz) {
        std::vector<T> data(sz);
        moveTo(pos * (sizeof(T) * sz));
        read(reinterpret_cast<char*>(data.data()), sizeof(T), sz);
        return data;
    }
};


template <typename T, size_t S>
/**
 * @brief The IdxReader<std::array<T, S> > class for data arrays
 */
class IdxRandomReader<std::array<T, S>> : public IdxRandomReaderBase {
public:
    /**
     * @brief IdxReader
     * @param name as above
     * @param iobufsz as above
     */
    IdxRandomReader(const std::string& name,
                    unsigned seed =
                #ifndef RANDOM_SEED
                    static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
                #else
                    RANDOM_SEED
                #endif
                    , unsigned iobufsz = DefaultIobufSzRandom) : IdxRandomReaderBase(name, seed, iobufsz) {}
    /**
     * @brief next
     * @return array of data.  see dimension(0) how many times to be done -  then other dimensions will defines what would be the size.
     * Except it is expected that user know data dimensions before hand as array size is give compile time.
     */
    std::array<T, S> next(size_t pos) {
        std::array<T, S> data;
        moveTo(pos * S * sizeof(T));
        read(reinterpret_cast<char*>(data.data()),  sizeof(T), S);
        return data;
    }
private:
    std::default_random_engine m_gen;
};

}


#endif // IDXREADER_H
