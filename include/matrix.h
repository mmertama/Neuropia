#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

#include <map>
#include <memory>

#include <thread>
#include <sstream>

#ifdef CHECK_VALUES
#define VALIDATE(x) (matrix_assert(!(std::isnan(x) || std::isinf(x))))
#else
#define VALIDATE(x)
#endif

#if !(defined(NDEBUG) && defined(_DEBUG))
#define matrix_assert(condition) ((void)0)
#else
#include <cassert>
#define matrix_assert assert
#endif

namespace Neuropia {

#ifndef STD_ALLOCATOR
/*
 * Per Thread based Multipool allocator.
 * Matrix operations do lot of alloc/dealloc and that is an issue
 * when having multiple threads using single memory std::allocator.
 *
 * However there is only pretty limited amount of different
 * Matrix sizes used (in Neuropia < 10) I have a pool for each
 * size. To avoid any conflict there is own pool for eaach thread.
 *
*/
template <class T>
class MatrixAllocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
private:
    struct Pool {
        Pool() = default;
        Pool(const Pool&) = delete;
        Pool(Pool&&) = delete;
        struct Node {
            Node* next;
            T* block;
        };
        std::map<size_t, Node*> nodes = {};
        ~Pool() {
            std::allocator<char> byteAllocator;
            for(auto it = nodes.begin(); it != nodes.end(); it++) {
                const auto size = it->first;
                auto n = it->second;
                while(n) {
                    auto next = n->next;
                    byteAllocator.deallocate(reinterpret_cast<char*>(n), size * sizeof(T) + sizeof(void*));
                    n = next;
                }
            }
        }
    };
public:
    MatrixAllocator() {}
    template <class U> MatrixAllocator(MatrixAllocator<U> const&) noexcept {}
    T* allocate(size_t size) {
        const auto it = m_pool.nodes.find(size);
        if(it == m_pool.nodes.end() || it->second == nullptr) {
            std::allocator<char> byteAlloc;
            auto bytes = byteAlloc.allocate(sizeof(T) * size + sizeof(void*));
            return reinterpret_cast<T*>(bytes + sizeof(void*));
        }

        auto node = it->second;
        auto next = node->next;
        it->second = next;
        return node->block;

    }
    void deallocate(T* ptr, size_t size) {
        auto node = reinterpret_cast<typename MatrixAllocator<T>::Pool::Node*>(reinterpret_cast<char*>(ptr) - sizeof(void*));
        node->block = ptr;
        const auto it = m_pool.nodes.find(size);
        if(it == m_pool.nodes.end()) {
            m_pool.nodes.emplace(size, node);
            node->next = nullptr;
        } else {
            node->next = it->second;
            it->second = node;
        }
    }
private:
    thread_local static Pool m_pool;
};

template <class T>
thread_local typename MatrixAllocator<T>::Pool MatrixAllocator<T>::m_pool;

template <class T, class U>
bool operator==(MatrixAllocator<T> const&, MatrixAllocator<U> const&) noexcept {
    return true;
}

template <class T, class U>
bool operator!=(MatrixAllocator<T> const& x, MatrixAllocator<U> const& y) noexcept {
    return !(x == y);
}
#endif

template <typename T>
class Matrix {
#ifndef STD_ALLOCATOR
    typedef std::vector<T, MatrixAllocator<T>> MatrixData;
public:
    typedef typename std::vector<T, MatrixAllocator<T>>::size_type index_type;
#else
    typedef std::vector<T> MatrixData;
public:
    typedef typename std::vector<T>::size_type index_type;
#endif
public:
    enum class VecDir {row, col};
    Matrix(index_type c, index_type r) noexcept : m_data(c * r), m_colSize(c) {
        if(c > 0 && r > 0) {
            operator()(0, 0) = std::numeric_limits<T>::infinity();
        }
    }
    Matrix(Matrix&& other) noexcept = default;
    Matrix(const Matrix& other) = delete;
    Matrix() = default;
    Matrix& operator=(const Matrix& other) = default;
    ~Matrix() = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    bool isValid() const noexcept {
        return !(m_colSize <= 0 || rows() <= 0 || cols() <= 0 || std::isinf(operator()(0, 0)) || std::isnan(operator()(0, 0)));
    }

    Matrix(std::initializer_list<std::initializer_list<T>> m) : Matrix(m.size() > 0 ? m.begin()->size() : 0, m.size()) {
        int j = 0;
        for(const auto& row : m) {
            int i = 0;
            for(const auto& v : row) {
                operator()(i, j) = v;
                ++i;
            }
            ++j;
        }
    }

    static Matrix<T> zero(index_type c, index_type r) noexcept {
        Matrix<T> m(c, r);
        m.set(0);
        return m;
    }

    void set(T value = 0) noexcept {
        for(auto  j = 0; j < rows(); j++) {
            for(auto  i = 0; i < cols(); i++) {
                operator()(i, j) = value;
            }
        }
    }

    void randomize(T min = 0, T max = static_cast<T>(1.0)) {
        unsigned seed =
#ifndef RANDOM_SEED
                static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
                RANDOM_SEED
#endif
                ;
        std::default_random_engine rd(seed);  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(min, max);

        for(auto  j = 0; j < rows(); j++) {
            for(auto  i = 0; i < cols(); i++) {
                operator()(i, j) =
                    dis(gen);
            }
        }
    }

    inline T& operator()(index_type c, index_type r) noexcept { return m_data[r * m_colSize + c];}

    inline T operator()(index_type c, index_type r) const noexcept { return m_data[r * m_colSize + c];}

    inline index_type rows() const noexcept {return m_data.size() / m_colSize;}

    inline index_type cols() const noexcept {return m_colSize;}

    const MatrixData& data() const noexcept {return m_data;}

    Matrix concatCols(const Matrix& a) const {
        matrix_assert(a.rows() == rows());
        Matrix m(a.cols() + cols(), rows());
        for(index_type i = 0; i < rows(); i++) {
            auto it = m.m_data[i].begin();
            std::copy(m_data[i].begin(), m_data[i].end(), it);
            std::advance(it, std::distance(m_data[i].begin(), m_data[i].end()));
            std::copy(a.m_data[i].begin(), a.m_data[i].end(), it);
        }
        return m;
    }

    Matrix concatRows(const Matrix& a) const {
        matrix_assert(a.cols() == cols());
        Matrix m(a.cols(), a.rows() + rows());

        for(index_type i = 0; i < rows(); i++) {
            std::copy(m_data[i].begin(), m_data[i].end(), m.m_data[i].begin());
        }
        for(index_type i = 0; i < a.rows(); i++) {
            std::copy(a.m_data[i].begin(), a.m_data[i].end(), m.m_data[i + rows()].begin());
        }
        return m;
    }

    Matrix copy(index_type c1, index_type r1, index_type c2, index_type r2) const {
        Matrix m(c2 - c1 + 1, r2 - r1 + 1);
        auto row = m.m_data.begin();
        for(index_type i = r1 ; i <=  r2; i++) {
            auto start = m_data[i].begin();
            std::advance(start, c1);
            std::copy_n(start, c2 - c1 + 1, row->begin());
            ++row;
        }
        return m;
    }

    Matrix row(index_type rowIndex, index_type start, index_type end) const {
        return copy(start, rowIndex, end, rowIndex);
    }

    Matrix col(index_type colIndex, index_type start, index_type end) const {
        return copy(colIndex, start, colIndex, end);
    }

    static Matrix map(const Matrix& a, const Matrix& b, std::function<T(const T&, const T&)> f) noexcept {
        matrix_assert(a.cols() == b.cols());
        matrix_assert(a.rows() == b.rows());
        Matrix m(a.cols(), a.rows());
        for(auto j = 0U; j < a.rows() ; j++) {
            for(auto i = 0U; i < a.cols() ; i++) {
                m(i, j) = f(a(i, j), b(i, j));
            }
        }
        return m;
    }

    Matrix map(std::function<T(const T&)> f) const noexcept {
        Matrix m(cols(), rows());
        for(auto j = 0U; j < rows() ; j++) {
            for(auto i = 0U; i < cols() ; i++) {
                m(i, j) = f(operator()(i, j));
            }
        }
        return m;
    }

    void mapThis(const Matrix& b, std::function<T(const T&, const T&)> f) noexcept {
        matrix_assert(cols() == b.cols());
        matrix_assert(rows() == b.rows());
        for(auto j = 0U; j < rows() ; j++) {
            for(auto i = 0U; i < cols() ; i++) {
                operator()(i, j) = f(operator()(i, j), b(i, j));
            }
        }
    }

    void mapThis(std::function<T(const T&)> f) noexcept {
        for(auto j = 0U; j < rows() ; j++) {
            for(auto i = 0U; i < cols() ; i++) {
                operator()(i, j) = f(operator()(i, j));
            }
        }
    }

    enum {ASVECTOR = -1};

    template <typename C>
    static Matrix<T> fromIterator(typename C::const_iterator begin,
                                  typename C::const_iterator end,
                                  int colLen = ASVECTOR,
    std::function<T(decltype(*begin)) > adaptor = [](const auto& v) noexcept {return v;}) {
        auto it = begin;
        const auto len = std::distance(begin, end);
        Matrix<T> m(
            static_cast<index_type>(colLen <= 0 ? len : colLen),
            static_cast<index_type>(colLen <= 0 ? 1 : len / colLen));
        if(len > 0) {
            for(auto r = 0U; r < m.rows(); r++) {
                for(auto c = 0U; c < m.cols(); c++) {
                    m(c, r) = adaptor(*it);
                    ++it;
                }
            }
        }
        return m;
    }

    template <typename C>
    static Matrix<T> fromArray(const C& array, VecDir row = VecDir::col) {
        return fromIterator<C>(array.begin(), array.end(), static_cast<int>(row == VecDir::row ? 1U : array.size()));
    }

    template <typename C>
    static Matrix<T> fromIterator(typename C::const_iterator begin,
                                  typename C::const_iterator end,
                                  index_type colLen,
                                  std::function<T(decltype(*begin), index_type c) > adaptor) {
        Matrix<T> m(colLen, static_cast<index_type>(std::distance(begin, end)));
        neuropia_assert_always(begin <= end, "Invalid");
        for(auto it = begin; it != end; ++it) {
            const auto r = static_cast<index_type>(std::distance(begin, it));
            for(index_type c = 0; c < colLen; ++c) {
                m(c, r) = adaptor(*it, c);
            }
        }
        return m;
    }

    template <typename C>
    static Matrix<T> fromArray(const C& array, index_type colLen,
                               std::function<T(decltype(*(array.begin())), index_type c) > adaptor) {
        return fromIterator<C>(array.begin(), array.end(), static_cast<index_type>(colLen), adaptor);
    }

    static Matrix<T> multiply(const Matrix<T>& m1, const Matrix<T>& m2) noexcept {
        return m1.multiply(m2);
    }

    template <typename R>
    R reduce(const R& init, std::function<R(const R&, const T&)> f) const noexcept {
        R out = init;
        for(auto j = 0U; j < rows() ; j++) {
            for(auto i = 0U; i < cols() ; i++) {
                out = f(out, operator()(i, j));
            }
        }
        return out;
    }

    T norm() const noexcept {
        return std::sqrt(reduce<T>(0.0,  [](const T a, const T p) {return a + static_cast<T>(p * p);}));
    }

    Matrix transpose() const noexcept {
        Matrix m(rows(), cols());
        for(auto j = 0U; j < rows() ; j++) {
            for(auto i = 0U; i < cols() ; i++) {
                m(j, i) = operator()(i, j);
            }
        }
        return m;
    }

    /*Naive implementation*/
    Matrix multiply(const Matrix& other) const noexcept {
        matrix_assert(cols() == other.rows());
        Matrix m(other.cols(), rows());
        for(auto j = 0U; j < m.rows(); j++) {
            for(auto i = 0U; i < m.cols(); i++) {
                T product = 0;
                for(auto k = 0U; k < cols(); k++) {
                    product += operator()(k, j) * other(i, k);
                }
                m(i, j) = product;
            }
        }
        return m;
    }

    /*Not the most effiecent implementation as algorithm wanna do sort in place first*/
    Matrix uniqueRows() const {
        auto m = *this;
        m.makeUnique();
        return m;
    }

    std::vector<T> toVector(VecDir row = VecDir::col, int index = 0) const {
        const int sz = row == Matrix<T>::VecDir::row ? cols() : rows();
        //issue when compiling, I have to explicitly define types first
        const std::function<T(int)> f1 = [this, index](int i) {
            return operator()(index, i);
        };
        const std::function<T(int)> f2 = [this, index](int i)  {
            return operator()(i, index);
        };
        const auto f = row == VecDir::row ?  f1 : f2 ;
        std::vector<T> vec(sz);
        for(int i = 0; i <  sz; i++) {
            vec[i] = f(i);
        }
        return vec;
    }

    void operator+=(const Matrix<T>& other) noexcept {
        return Matrix<T>::mapThis(other, [](const T & a, const T & b) {return a + b;});
    }

    void operator*=(const Matrix<T>& other) noexcept {
        return Matrix<T>::mapThis(other, [](const T & a, const T & b) {return a * b;});
    }

    void operator-=(const Matrix<T>& other) noexcept {
        return Matrix<T>::mapThis(other, [](const T & a, const T & b) {return a - b;});
    }

    void operator/=(const Matrix<T>& other) noexcept {
        return Matrix<T>::mapThis(other, [](const T & a, const T & b) {return a / b;});
    }


    friend std::ostream& operator<<(std::ostream& output, const Matrix<T>& mat) {
        output << '[' << std::endl;
        if(mat.isValid()) {
            for(int j = 0; j < mat.rows(); j++) {
                for(int i = 0; i < mat.cols(); i++) {
                    output << mat(i, j) << " ";
                }
                output << std::endl;
            }
        }
        output << ']';
        return output;
    }

    friend Matrix<T> operator-(const Matrix<T>& m1, const Matrix<T>& m2) noexcept {
        return Matrix<T>::map(m1, m2, [](const T & a, const T & b) noexcept {return a - b;});
    }

    friend Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2) noexcept {
        return Matrix<T>::map(m1, m2, [](const T & a, const T & b) noexcept {return a + b;});
    }

    friend Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) noexcept {
        return Matrix<T>::map(m1, m2, [](const T & a, const T & b) noexcept {return a * b;});
    }

    friend Matrix<T> operator*(const Matrix<T>& m1, const T& v) noexcept {
        return m1.map([v](const T & a) noexcept {return a * v;});
    }

    friend  Matrix<T> operator*(const T& v, const Matrix<T>& m1) noexcept {
        return m1.map([v](const T & a) noexcept {return a * v;});
    }


private:
    void makeUnique() {
        const auto compare = [](const std::vector<T>& a, const std::vector<T>& b)-> int {
            matrix_assert(a.size() == b.size());
            for(index_type i = 0; i < a.size(); i++) {
                const auto av = a.at(i);
                const auto bv = b.at(i);
                if(av < bv) { return -1; }
                if(av > bv) { return 1; }
            }
            return 0;
        };
        std::sort(m_data.begin(), m_data.end(), [compare](const std::vector<T>& a, const std::vector<T>& b) {
            return compare(a, b) < 0;
        });
        const auto it = std::unique(m_data.begin(), m_data.end(), [compare](const std::vector<T>& a, const std::vector<T>& b) {
            return compare(a, b) == 0;
        });
        m_data.erase(it, m_data.end());
    }
private:
    MatrixData m_data = {};
    index_type m_colSize = 0;
};


}

#endif // MATRIX_H
