#include <chrono>       // std::chrono::system_clock
#include <fstream>
#include <algorithm>
#include "neuropia.h"
#include "utils.h"
#include "matrix.h"

using namespace Neuropia;

static char mapGrey(unsigned char m) {
    constexpr char greys[] = " .:-=+*#%@";
    const auto p = m / 26;
    neuropia_assert(p >= 0 && p < 10);
    return greys[9 - p];
}

void Neuropia::printimage(const unsigned char* c)  {
    int p = 0;
    for(int j = 0; j < 28; j++) {
        for(int i = 0; i < 28; i++) {
            std::cout << mapGrey(c[p]);
            ++p;
        }
        std::cout << std::endl;
    }
}


void Neuropia::printVerify(const std::tuple<size_t, size_t>& result, const std::string& txt) {
    std::cout << txt << ", rate:" << 100.0 * (static_cast<double>(std::get<0>(result)) / static_cast<double>(std::get<1>(result)))
          << "%, found:" << std::get<0>(result)
          << " of " << std::get<1>(result)<< std::endl;
}

size_t Neuropia::iterator(size_t iterations, const std::function<bool ()>& f) {
    return iterator(iterations, static_cast<std::function<bool (size_t)>>([f](size_t)->bool{return f();}));
}

size_t Neuropia::iterator(size_t iterations, const std::function<bool (size_t)>& f) {
    for(size_t i = 0U; i < iterations ; i++) {
        if(!f(i)) {
            return i;
        }
    }
    return iterations;
}

void Neuropia::timed(const std::function<void ()>& f, const std::string& label) {
    const auto start = std::chrono::high_resolution_clock::now();
    f();
    const auto stop = std::chrono::high_resolution_clock::now();
    std::cout << (!label.empty() ? label + " " : "")
              << "timed:" << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()
              << "." <<  std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
              - std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() * 1000000 << std::endl;
}

void Neuropia::save(const std::string& filename, const Neuropia::Layer& network, const std::unordered_map<std::string, std::string>& map) {
    std::ofstream str;
    str.open(filename, std::ios::out | std::ios::binary);
    if(str.is_open()) {
        network.save(str, map);
    }
    str.close();
}

void  Neuropia::save(const std::string& filename, const std::vector<Layer>& ensembles) {
    std::ofstream str;
    str.open(filename, std::ios::out | std::ios::binary);
    if(str.is_open()) {
        for (const auto& n : ensembles) {
            n.save(str);
        }
    }
    str.close();
}


std::vector<Layer> Neuropia::loadEnsemble(const std::string& filename) {
    std::vector<Layer> ensembles;
    std::ifstream str;
    str.open(filename, std::ios::in | std::ios::binary);
    while(str.is_open() && !str.eof()) {
        ensembles.emplace_back(Layer(str));
    }
    str.close();
    return ensembles;
}

Layer Neuropia::load(const std::string& filename, const std::function<void (const std::unordered_map<std::string, std::string>&) >& metareader ) {
    Neuropia::Layer network;
    std::ifstream str;
    str.open(filename, std::ios::out | std::ios::binary);
    if(str.is_open()) {
        network = Neuropia::Layer(str, metareader);
    } else std::cerr << "filename " << filename << " cannot be opened" << std::endl;
    str.close();
    return network;
}


void Neuropia::debug(const Neuropia::Layer& layer, std::ostream& out, const std::set<int>& excluded) {
    auto l = &layer;
    int count = 0;
    while(l) {
        ++count;
        if(excluded.find(count) == excluded.end()) {
            int neuron = 0;
            for(const auto&n : *l) {
                ++neuron;
                double avg = 0;
                for(auto i = 0U; i < n.size(); i++)
                    avg += n.weight_d(i);
                avg /= static_cast<double>(n.size());
                out << "layer " << count << ", neuron:" << neuron << ", weights avg:"<< avg << ", bias:" << n.bias() << ", function:" << l->activationFunction().name() << ", active:" << n.isActive() << std::endl;
            }
        }
        l = l->get(1);
    }
}

std::string Neuropia::absPath(const std::string& root, const std::string& relativePath) {
    return root + "/" + relativePath;
}
