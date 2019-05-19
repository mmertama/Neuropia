#ifndef ARGPARSE_H
#define ARGPARSE_H

#include <unordered_map>
#include <vector>
#include <tuple>
#include <set>
#include <iostream>


/**
 * @brief The ArgParse class
 *
 * Yet another implementation of Python style ArgumentParser
 *
 */
class ArgParse {
public:
    ArgParse& addOpt(char shortName, std::string longName = "", bool hasValue = false, std::string defaultValue = "");
    bool set(int argc, char** argv, char shortOpt = '-', const std::string& longOpt = "--");
    std::unordered_map<std::string, std::string> options() const;
    std::vector<std::string> parameters() const;
    bool hasOption(const std::string& name) const;
    bool hasParameter(const std::string& name) const;
    bool hasOption(char name) const;
    std::string param(size_t index) const;
    std::string option(char k) const;
    std::string option(const std::string& k) const;
    size_t paramCount() const;
protected:
    std::vector<std::string> m_parameters;
    std::unordered_map<std::string, std::tuple<bool, bool, std::string>> m_options;
    std::unordered_map<std::string, std::string> m_optionsAlias;
};

#endif // ARGPARSE_H
