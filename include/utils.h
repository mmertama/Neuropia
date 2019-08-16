#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <string>
#include <iostream>
#include <set>
#include <iomanip>
#include <vector>
#include <unordered_map>

namespace Neuropia {
class Layer;

void timed(const std::function<void ()>& f, const std::string& label = "");

size_t iterator(size_t iterations, const std::function<bool (size_t it)>& f);

size_t iterator(size_t iterations, const std::function<bool ()>& f);

void save(const std::string& filename, const Layer& network, const std::unordered_map<std::string, std::string>& = {});

void save(const std::string& filename, const std::vector<Layer>& ensembles);

std::vector<Layer> loadEnsemble(const std::string& filename);

std::tuple<Neuropia::Layer, bool, std::unordered_map<std::string, std::string>> load(const std::string& filename);

void debug(const Layer& network, std::ostream& strm = std::cout, const std::set<int>& excluded = {});

void printimage(const unsigned char* c);

std::string absPath(const std::string& root, const std::string& relativePath);

void printVerify(const std::tuple<size_t, size_t>& result, const std::string& txt);

template <typename T1, typename T2>
void persentage(T1 fraction, T2 total, const std::string& extra = "") {
        const auto f = 100.0 * (static_cast<double>(fraction) / static_cast<double>(total));
        std::cout << "\r" << std::fixed << std::setprecision(3) << f << '%' << extra << std::flush;
}
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
