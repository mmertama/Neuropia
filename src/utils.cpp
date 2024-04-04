#include <chrono>       // std::chrono::system_clock
#include <fstream>
#include <algorithm>
#include <random>
#include "neuropia.h"
#include "utils.h"
#include "matrix.h"
#include "verify.h"

using namespace Neuropia;

bool fatal(const char* t, const char* f, int line, const char* file) {
    std::cerr << "Assert:" << t << " in line " << line << " at " << f << "  in "<< file << "." << std::endl;
    std::abort();
}

static char mapGrey(unsigned char m) {
    constexpr char greys[] = " .:-=+*#%@";
    const auto p = m / 26;
    neuropia_assert(p >= 0 && p < 10);
    return greys[9 - p];
}

void Neuropia::printimage(const unsigned char* c, int width, int height)  {
    int p = 0;
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            std::cout << mapGrey(c[p]);
            ++p;
        }
        std::cout << std::endl;
    }
}


void Neuropia::printVerify(const Neuropia::VerifyResult& result, std::string_view txt) {
    std::cout.flush();
    std::cout << txt << ", rate:" <<  std::get<1>(result) * 100.
          << "%, found:" << std::get<0>(result)
          << " of " << std::get<2>(result) << std::endl;
    std::cout.flush();      
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
    std::cout << std::endl; // flush!
    std::cout << (!label.empty() ? label + " " : "")
              << "timed:" << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()
              << "." <<  std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
              - std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() * 1000000 << std::endl;      
}

void Neuropia::save(const std::string& filename, const Neuropia::Layer& network, const std::unordered_map<std::string, std::string>& map, Neuropia::SaveType savetype) {
    std::ofstream str;
    str.open(filename, std::ios::out | std::ios::binary);
    if(str.is_open()) {
        network.save(str, map, savetype);
    }
    str.close();
}

void  Neuropia::save(const std::string& filename, const std::vector<Layer>& ensembles, Neuropia::SaveType savetype) {
    std::ofstream str;
    str.open(filename, std::ios::out | std::ios::binary);
    if(str.is_open()) {
        for (const auto& n : ensembles) {
            n.save(str, {}, savetype);
        }
    }
    str.close();
}


std::vector<Layer> Neuropia::loadEnsemble(const std::string& filename) {
    std::vector<Layer> ensembles;
    std::ifstream str;
    str.open(filename, std::ios::in | std::ios::binary);
    while(str.is_open() && !str.eof()) {
        Layer layer;
        const auto m = layer.load(str);
        if(m)
            ensembles.emplace_back(layer);
    }
    str.close();
    return ensembles;
}

std::optional<std::tuple<Neuropia::Layer, std::unordered_map<std::string, std::string>>> Neuropia::load(const std::string& filename) {
    Neuropia::Layer network;
    std::ifstream str;
    str.open(filename, std::ios::out | std::ios::binary);
    if(str.is_open()) {
        const auto map = network.load(str);
        if(!map) {
            std::cerr << "filename " << filename << " is corrupted" << std::endl;
            return std::nullopt;
        }
        return std::make_tuple(network, *map);
    } 
    std::cerr << "filename " << filename << " cannot be opened" << std::endl;
    return std::nullopt;
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
                NeuronType avg = 0;
                for(auto i = 0U; i < n.size(); i++)
                    avg += n.weight_d(i);
                avg /= static_cast<NeuronType>(n.size());
                out << "layer " << count << ", neuron:" << neuron << ", weights avg:"<< avg << ", bias:" << n.bias() << ", function:" << l->activationFunction().name() << ", active:" << n.isActive() << std::endl;
            }
        }
        l = l->get(1);
    }
}

std::string Neuropia::absPath(const std::string& root, const std::string& relativePath) {
    return root.empty() || (!relativePath.empty() && relativePath.front() == '/') ?
        relativePath : root + "/" + relativePath;
}

Random::Random(unsigned seed) : m_gen(seed) {}

Random::Random() : m_gen(
#ifndef RANDOM_SEED
    static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
    RANDOM_SEED
#endif
) {}

size_t Random::random(size_t atop) {
    neuropia_assert(atop > 0);
    return (m_gen() % atop);
    }

std::string_view Neuropia::to_string(Neuropia::SaveType st) {
    static const std::unordered_map<Neuropia::SaveType, std::string_view> map{
        {Neuropia::SaveType::SameAsNeuronType, "SameAsNeuronType"}, 
        {Neuropia::SaveType::Double, "Double"}, 
        {Neuropia::SaveType::Float, "Float"}, 
        {Neuropia::SaveType::LongDouble, "LongDouble"}};
    return map.at(st);    
}

bool Neuropia::isnumber(std::string_view s, bool allow_negative, std::optional<char> digit_sep) {
    if(s.empty())
        return false;
    auto begin = s.begin();
    if(!((*begin == '-' && allow_negative) || std::isdigit(*begin) || (digit_sep && *begin == *digit_sep)))
        return false;
    if(digit_sep && *begin == *digit_sep)
        digit_sep = std::nullopt;    
    ++begin;
#if !defined(__clang__) && !defined(__EMSCRIPTEN__) && (defined(__GNUC__) || defined(__GNUG__))   
#pragma GCC diagnostic push  // GCC bug?
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif       
    return std::find_if(begin, s.end(), [&digit_sep](auto c) mutable {
        if(digit_sep && c == *digit_sep) {
            digit_sep = std::nullopt;
            return true;
        }     
            return !std::isdigit(c);
        }) == s.end();
#if defined(__GNUC__) || defined(__GNUG__)            
#pragma GCC diagnostic push      
#endif  
}


std::string_view Neuropia::to_string(bool value) {
    static constexpr auto TRUE = "true";
    static constexpr auto FALSE = "false";
    return value ? TRUE : FALSE;
}