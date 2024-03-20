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

namespace Neuropia {

void timed(const std::function<void ()>& f, const std::string& label = "");

size_t iterator(size_t iterations, const std::function<bool (size_t it)>& f);

size_t iterator(size_t iterations, const std::function<bool ()>& f);

void save(const std::string& filename, const Layer& network, const std::unordered_map<std::string, std::string>& = {}, SaveType savetype = SaveType::SameAsNeuronType);

void save(const std::string& filename, const std::vector<Layer>& ensembles, SaveType savetype = SaveType::SameAsNeuronType);

std::vector<Layer> loadEnsemble(const std::string& filename);

std::optional<std::tuple<Neuropia::Layer, std::unordered_map<std::string, std::string>>> load(const std::string& filename);

void debug(const Layer& network, std::ostream& strm = std::cout, const std::set<int>& excluded = {});

void printimage(const unsigned char* c, int width, int height);

std::string absPath(const std::string& root, const std::string& relativePath);

void printVerify(const std::tuple<size_t, size_t>& result, const std::string& txt);

template <typename T1, typename T2>
void percentage(T1 fraction, T2 total, const std::string& extra = "") {
        const auto f = 100.0 * (static_cast<double>(fraction + 1) / static_cast<double>(total));
        std::cout << "\r" << std::fixed << std::setprecision(3) << f << '%' << extra << std::flush;
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
