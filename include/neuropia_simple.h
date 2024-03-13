#ifndef NEUROPIA_SIMPLE_H
#define NEUROPIA_SIMPLE_H

#include <string>
#include <limits>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include "neuropia.h"


/**
 * @brief Wraps Neuropia behind a simple interface.
 */
namespace NeuropiaSimple {


class NeuropiaEnv;
using NeuropiaPtr = std::shared_ptr<NeuropiaEnv>;
using ParamType = std::map<std::string, std::vector<std::string>>;



enum class TrainType {
    Basic,
    Evolutional,
    Parallel
};



/**
 * @brief Create an environent
 * @param root file path root
 * @return NeuropiaPtr 
 */
NeuropiaPtr create(const std::string& root);

/**
 * @brief Free up the env
 * 
 * @param env 
 */
void free(NeuropiaPtr env);

/**
 * @brief Feeds a input for the input layer, return the output
 * 
 * @param env 
 * @param input 
 * @return std::vector<NeuronType> 
 */
std::vector<Neuropia::NeuronType> feed(NeuropiaPtr env, const std::vector<Neuropia::NeuronType>& input);

/**
 * @brief Set the Param values, see parameter names from params.h or default.h, where te default parameters defined
 * 
 * @param env 
 * @param name 
 * @param value 
 * @return true 
 * @return false 
 */
bool setParam(const NeuropiaPtr& env, const std::string& name, const std::string& value);

/**
 * @brief Test if a a given paramter is valid. 
 * 
 * @param env 
 * @param name 
 * @param value 
 * @return true 
 * @return false 
 */
bool isValid(const NeuropiaPtr& env, const std::string& name, const std::string& value);

/**
 * @brief Get current parameters.
 * 
 * @param env 
 * @return ParamType 
 */
ParamType params(const NeuropiaPtr& env);


/**
 * @brief Train network.
 * 
 * @param env 
 * @param type 
 * @return true 
 * @return false 
 */
bool train(const NeuropiaPtr& env, TrainType type);

/**
 * @brief Store network to a file.
 * 
 * @param env 
 * @param filename 
 */
void save(const NeuropiaPtr& env, const std::string& filename, Neuropia::SaveType savetype = Neuropia::SaveType::SameAsNeuronType);

/**
 * @brief Load network from a file.
 * 
 * @param env 
 * @param filename 
 * @return true 
 * @return false 
 */
bool load(const NeuropiaPtr& env, const std::string& filename);

/**
 * @brief Calculates test material over the network to comparative value of network match accuracy.
 * 
 * @param env 
 * @return int 
 */
int verify(const NeuropiaPtr& env);

/**
 * @brief Set the logging output handler.
 * 
 * @param env 
 * @param cb 
 */
void setLogger(const NeuropiaPtr& env, std::function<void (const std::string&) > cb);

}

#endif // NEUROPIA_SIMPLE_H
