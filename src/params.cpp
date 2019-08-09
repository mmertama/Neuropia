
#include <limits>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <iomanip>
#include <sstream>
#include "params.h"

using namespace Neuropia;

constexpr char Neuropia::Params::Execute[];
constexpr char Neuropia::Params::File[];
constexpr char Neuropia::Params::String[];
constexpr char Neuropia::Params::Int[];
constexpr char Neuropia::Params::Real[];
constexpr char Neuropia::Params::Bool[];

std::vector<std::string> Params::split(const std::string& str) {
       std::vector<std::string> top;
       std::istringstream f(str);
       std::string s;
       while(std::getline(f, s, ',')) {
           top.push_back(s);
       }
       return top;
   }

Params::Params(const std::initializer_list<std::tuple<std::string, std::string, std::string>>& lst) : m_data(lst) {
    push_back({Execute, "", Params::String}); //for set current pseudo command, to enable print current before executed
}

bool Params::contains(const std::string& k) const {
    for(const auto& p : (*this)) {
        if(std::get<0>(p) == k) {
            return true;
        }
    }
    return false;
}

std::string Params::operator[](const std::string& k) const {
    for(const auto& p : (*this)) {
        if(std::get<0>(p) == k) {
            return std::get<1>(p);
        }
    }
    return "";
}

bool Params::isValid(const std::string& k, const std::string& v) const {
    for(auto& p : (*this)) {
        if(std::get<0>(p) == k) {
            const std::regex re(std::get<2>(p));
                return std::regex_match(v, re);
        }
    }
    return false;
}

bool Params::set(const std::string& k, const std::string& v) {
    for(auto& p : (*this)) {
        if(std::get<0>(p) == k) {
            const std::regex re(std::get<2>(p));
            if(std::regex_match(v, re)) {
                std::get<1>(p) = v;
                return true;
            }
            return false;
        }
    }
    return false;
}

bool Params::isZero(double v) {
    return v < std::numeric_limits<double>::epsilon() &&  v > -std::numeric_limits<double>::epsilon();
}

bool Params::isZero(const std::string& key) const {
    return isZero(std::stod(operator[](key)));
}

bool Params::boolean(const std::string& key) const {
    auto v = operator[](key);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "false" || v == "0" || v.empty());
}

int Params::integer(const std::string& key) const {
    return std::stoi(operator[](key));
}

unsigned Params::uinteger(const std::string& key) const {
    return static_cast<unsigned>(std::stoul(operator[](key)));
}

double Params::real(const std::string& key) const {
    return std::stod(operator[](key));
}

std::string Params::formatPrint(const std::string& line) const {
    std::string out;
    for(auto it = line.begin(); it < line.end(); ++it) {
        if(*it == '$') {
            ++it;
            if(*it == '$') {
                out += '$';
            } else {
                const auto start = static_cast<unsigned>(std::distance(line.begin(), it));
                const auto end = line.find_first_of(' ', start);
                const std::string token(line, start, std::min(line.length(), end) - start);
                it += static_cast<int>(token.length()) - 1; //$ char
                for(const auto& p : *this) {
                    if(std::get<0>(p) == token) {
                        out += std::get<1>(p);
                        break;
                    }
                }
            }
        } else {
            out += *it;
        }
    }
    return out;
}


bool Params::readTask(const std::string& filename,
              const std::unordered_map<std::string, std::function<void(const std::string& root)>>& tasks,
              const std::string& root) {
    std::ifstream is(filename);
    if(!is.is_open()) {
        return false;
    }
    int lineNo = 1;
    std::string line;
    while(!is.eof()) {
        std::string key;
        std::string value;
        is >> key;
        ++lineNo;
        if(key == "" || key[0] == '#') {
            std::string line;
            std::getline(is, line); //eat and forget line
            continue;
        }
        if(key == "print") {
            std::string line;
            std::getline(is, line);
            std::cout << formatPrint(line.erase(0, 1)) << std::endl;
            continue;
        }
        if(key == "run") {
            key = operator[](Params::Execute);
        }
        if(key == "exit") {
            break;
        }
        if(tasks.find(key) != tasks.end()) {
            tasks.at(key)(root);
            continue;
        }
        is >> value;
        if(!set(key, value)) {
            std::cerr << "Invalid key: " << key << ", near line " << lineNo << std::endl;
            return false;
        }
    }
    return true;
}

void Params::addHelp(const std::string& re, const std::string& text) {
    m_helps.emplace(re, text);
}

std::string Params::toType(const std::string &re) const {
    if(m_helps.find(re) != m_helps.end()) {
        return m_helps.at(re);
    }
    if(re == File) return "file";
    if(re == String) return "string";
    if(re == Int) return "interger number";
    if(re == Real) return "real number";
    if(re == Bool) return "true|false";
    const std::regex reg(R"(\((.*)\)\$)");
    std::smatch m;
    if(std::regex_match(re, m, reg)) {
       return m[1].str();
    }
    return re;
}

