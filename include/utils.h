#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <string>
#include <iostream>
#include <set>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <optional>
#include <string_view>
#include "neuropia.h"

bool fatal(const char* t, const char* f, int line, const char* file);

#define ASSERT(X) ((X) || fatal("Invalid", __FUNCTION__, __LINE__, __FILE__))
#define ASSERT_X(X, T) ((X) || fatal((T), __FUNCTION__, __LINE__, __FILE__))

namespace Neuropia {

void timed(const std::function<void ()>& f, const std::string& label = "");

size_t iterator(size_t iterations, const std::function<bool (size_t it)>& f);

size_t iterator(size_t iterations, const std::function<bool ()>& f);

void save(const std::string& filename, const Layer& network, const std::unordered_map<std::string, std::string>& = {}, SaveType saveType = SaveType::SameAsNeuronType);

void save(const std::string& filename, const std::vector<Layer>& ensembles, SaveType saveType = SaveType::SameAsNeuronType);

std::vector<Layer> loadEnsemble(const std::string& filename);

std::optional<std::tuple<Neuropia::Layer, std::unordered_map<std::string, std::string>>> load(const std::string& filename);

void debug(const Layer& network, std::ostream& strm = std::cout, const std::set<int>& excluded = {});

void printimage(const unsigned char* c, int width, int height);

std::string absPath(const std::string& root, const std::string& relativePath);

void printVerify(const Neuropia::VerifyResult& result, std::string_view txt);

template <typename T1, typename T2>
void percentage(T1 fraction, T2 total, const std::string& extra = "") {
        const auto f = 100.0 * (static_cast<double>(fraction) / static_cast<double>(total));
        std::cout << "\r" << std::fixed << std::setprecision(3) << f << '%' << extra << std::flush;
}


bool isnumber(std::string_view s, bool allow_negative = false, std::optional<char> digit_sep = std::nullopt);

template<typename IT, typename C>
void ordered_container(IT begin, C& order) {
    for(unsigned o = 0; o < order.size(); ++o)
        order[o] = std::make_pair(o, *(begin + o));
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b){return a.second > b.second;});    
}

template<typename IT, size_t SZ>
auto ordered(IT begin) {
    std::array<std::pair<unsigned, typename IT::value_type>, SZ> array;
    ordered_container(begin, array);
    return array;
}

template<typename IT>
auto ordered(IT begin, IT end) {
    std::vector<std::pair<unsigned, typename IT::value_type>> vector(static_cast<size_t>(std::distance(begin, end)));
    ordered_container(begin, vector);
    return vector;
}


class Random {
public:
    explicit Random(unsigned seed);
    Random();
    size_t random(size_t atop);
private:
    std::default_random_engine m_gen;
};

std::string_view to_string(Neuropia::SaveType st);
std::string_view to_string(bool value);

}


template <typename T>
std::ostream& operator << (std::ostream& strm, const std::vector<T>& values) {
    strm << '[';
    for(size_t i = 0; i < values.size() - 1; i++) {
        strm << values[i] << ", ";
    }
    strm << values[values.size() - 1];
    strm << ']';
    return strm;
}



#endif // UTILS_H
