#ifndef PARAMS_H
#define PARAMS_H

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace Neuropia {

class Params {
public:
    static constexpr char Execute[] = "executable";
    static constexpr char String[] = R"(.+)";
    static constexpr char Int[] = R"(\d+$)";
    static constexpr char Real[] = R"(\d+(\.(\d)*)?$)";
    static constexpr char Bool[] = R"((true|false)$)";
    Params(const std::initializer_list<std::tuple<std::string, std::string, std::string>>& lst);

    template<class P>
    void readParams(const P& argparse) {
        for(auto it = begin(); it != end(); ++it) {
            const auto index = static_cast<unsigned>(std::distance(begin(), it) + 2);
            if(index >= argparse.paramCount()) {
                break;
            }
            std::get<1>(*it) = argparse.param(index);
        }
    }

    static std::vector<std::string> split(const std::string& str);

    template <typename T>
    static std::vector<T> toVector(const std::string& str, std::function<T (const std::string&)> f) {
        std::vector<T> top;
        const auto l = split(str);
        std::transform(l.begin(), l.end(), std::back_inserter(top), f);
        return top;
    }

    bool contains(const std::string& k) const;
    std::string operator[](const std::string& k) const;
    static bool isZero(double v);
    bool isZero(const std::string& key) const;
    bool boolean(const std::string& key) const ;
    int integer(const std::string& key) const;
    unsigned uinteger(const std::string& key) const;
    double real(const std::string& key) const ;
    bool readTask(const std::string& filename,
                  const std::unordered_map<std::string, std::function<void(const std::string& root)>>& tasks,
                  const std::string& root);
    std::string toType(const std::string& re) const;
    void addHelp(const std::string& re, const std::string& text);
private:
    bool set(const std::string& k, const std::string& v);
    std::string formatPrint(const std::string& line) const;
private:
    std::unordered_map<std::string, std::string> m_helps;
    std::vector<std::tuple<std::string, std::string, std::string>> m_data;
public:
    decltype (m_data)::iterator begin() {return m_data.begin();}
    decltype (m_data)::iterator end() {return m_data.end();}
    decltype (m_data)::const_iterator begin() const {return m_data.begin();}
    decltype (m_data)::const_iterator end() const {return m_data.end();}
    void push_back(const decltype(m_data)::value_type& value) {m_data.push_back(value);}
};
}


#endif // PARAMS_H
