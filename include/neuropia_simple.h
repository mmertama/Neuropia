#ifndef NEUROPIA_SIMPLE_H
#define NEUROPIA_SIMPLE_H

#include <string>
#include <limits>
#include <vector>
#include <memory>
#include <map>


namespace NeuropiaSimple {


class NeuropiaEnv;
using NeuropiaPtr = std::shared_ptr<NeuropiaEnv>;
using ParamType = std::map<std::string, std::vector<std::string>>;
NeuropiaPtr create(const std::string& root);
void free(NeuropiaPtr env);


std::vector<double> feed(NeuropiaPtr env, const std::vector<double>& input);

bool setParam(NeuropiaPtr env, const std::string& name, const std::string& value);
bool isValid(NeuropiaPtr env, const std::string& name, const std::string& value);
ParamType params(NeuropiaPtr env);

enum class TrainType {
    Basic,
    Evolutional,
    Parallel
};


bool train(NeuropiaPtr env, TrainType type);

void save(NeuropiaPtr env, const std::string& filename);

bool load(NeuropiaPtr env, const std::string& filename);

int verify(NeuropiaPtr env);

void setLogger(NeuropiaPtr env, std::function<void (const std::string&) > cb);

}

#endif // NEUROPIA_SIMPLE_H
