#include "argparse.h"

ArgParse& ArgParse::addOpt(char shortName, std::string longName, bool hasValue, std::string defaultValue) {
    const std::string s(1, shortName);
    m_optionsAlias.emplace(s, longName.length() > 0 ? longName : s);
    m_options.emplace( m_optionsAlias[s], std::make_tuple(false, hasValue, defaultValue));
    return *this;
}

bool ArgParse::set(int argc, char** argv, char shortOpt, const std::string& longOpt) {
    auto isOpt = [shortOpt, longOpt, this](const std::string& param)->std::string{
        if(param[0] == shortOpt) {
            if(param.length() > 1) {
                const auto s = param.substr(1, 1);
                if(m_optionsAlias.find(s) != m_optionsAlias.end()) {
                    return m_optionsAlias[s];
                }
            }
        } else if(param.find(longOpt) == 0) {
            if(param.length() > longOpt.length()) {
                const auto s = param.substr(longOpt.length());
                if(m_optionsAlias.find(s) != m_optionsAlias.end()) {
                    m_optionsAlias[s];
                }
            }
        }
        return "";
    };
    for(auto i = 0; i < argc; i++) {
        const std::string param(argv[i]);
        const auto opt = isOpt(param);
        if(opt.empty()) {
            m_parameters.push_back(param);
        } else {
            auto& value = m_options[opt];
            std::get<0>(value) = true;
            if(std::get<1>(value)) {
                if(i < argc - 1 && isOpt(argv[i + 1]).empty()) {
                     std::get<2>(value) = argv[++i];
                } else {
                    return false; //param expected
                }
            } else {
                std::get<2>(value) = "true";
            }
        }
    }
    return true;
}

std::unordered_map<std::string, std::string> ArgParse::options() const {
    std::unordered_map<std::string, std::string> opt;
    for(const auto& v : m_options) {
        if(std::get<0>(v.second)) {
            opt.emplace(v.first, std::get<2>(v.second));
        }
    }
    return opt;
}

std::vector<std::string> ArgParse::parameters() const {
    return m_parameters;
}

bool ArgParse::hasOption(const std::string& name) const {
    const auto it = m_options.find(name);
    return it != m_options.end() && std::get<0>(it->second);
}


bool ArgParse::hasParameter(const std::string& name) const {
    for (const auto& v : m_parameters)
        if( v == name)
            return true;
    return false;
}

bool ArgParse::hasOption(char name) const {
     const auto it = m_optionsAlias.find(std::string(1, name));
     return it != m_optionsAlias.end() && hasOption(it->second);
}

size_t ArgParse::paramCount() const {
    return m_parameters.size();
}

std::string ArgParse::param(size_t index) const {
    return m_parameters.at(index);
}

std::string ArgParse::option(char k) const {
    const auto it = m_optionsAlias.find(std::string(1, k));
    return option(it->second);
}

std::string ArgParse::option(const std::string& k) const {
    const auto it = m_options.find(k);
    return std::get<2>(it->second);
}
