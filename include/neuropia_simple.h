#ifndef NEUROPIA_SIMPLE_H
#define NEUROPIA_SIMPLE_H

#include <string>
#include <limits>
#include <vector>
#include <memory>
#include <map>
#include <variant>
#include <functional>
#include "neuropia.h"


/**
 * @brief Wraps Neuropia behind a simple interface.
 */
namespace NeuropiaSimple {


class NeuropiaEnv;
using NeuropiaPtr = std::shared_ptr<NeuropiaEnv>;
using ParamMap = std::map<std::string, std::vector<std::string>>;



enum class TrainType {
    Basic,
    Evolutional,
    Parallel
};



/**
 * @brief Create an environment
 * @param root file path root
 * @return NeuropiaPtr 
 */
NeuropiaPtr create(const std::string& root = "");

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
 * @brief Set the Param values, see parameter names from params.h or default.h, where the default parameters defined
 * 
 * @param env 
 * @param name 
 * @param value 
 * @return true 
 * @return false 
 */
bool setParam(const NeuropiaPtr& env, const std::string& name, const std::string& value);
bool setParam(const NeuropiaPtr& env, const std::string& name, const std::variant<int, double, float, bool>& value);

/**
 * @brief Test if a a given parameter is valid. 
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
 * @return ParamMap
 */
ParamMap params(const NeuropiaPtr& env);


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
void save(const NeuropiaPtr& env, const std::string& filename, Neuropia::SaveType saveType = Neuropia::SaveType::SameAsNeuronType);

/**
 * @brief Load network from a file.
 * 
 * @param env 
 * @param filename 
 * @return true 
 * @return false 
 */
std::optional<Neuropia::Sizes> load(const NeuropiaPtr& env, const std::string& filename);

/**
 * @brief Calculates test material over the network to comparative value of network match accuracy.
 * 
 * @param env 
 * @param count, only read count first values from file - when using count verify that data is in random enough
 * @return int items correctly found from the training material
 */
int verify(const NeuropiaPtr& env, size_t count = std::numeric_limits<size_t>::max());

/**
 * @brief Set the logging output handler.
 * 
 * @param env 
 * @param call_back 
 */
void setLogger(const NeuropiaPtr& env, std::function<void (const std::string&) > call_back);

 /**
 * @brief Access to Neuropia network input layer
 * 
 * @return const Layer& 
 */
const Neuropia::Layer& network(const NeuropiaPtr& env);
}

#endif // NEUROPIA_SIMPLE_H
