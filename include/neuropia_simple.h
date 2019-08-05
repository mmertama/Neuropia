#ifndef NEUROPIA_SIMPLE_H
#define NEUROPIA_SIMPLE_H

#include <string>
#include <limits>
#include <vector>

class NeuropiaEnv;

NeuropiaEnv* createNeuropia(const std::string& root);
void free(NeuropiaEnv* env);

std::vector<double> feed(NeuropiaEnv* env, const std::vector<double>& input);

bool setParam(NeuropiaEnv* env, const std::string& name, const std::string& value);

bool trainSimple(NeuropiaEnv* env);
bool trainEvo(NeuropiaEnv* env);
bool trainParallel(NeuropiaEnv* env);

void save(NeuropiaEnv* env, const std::string& filename);

bool load(NeuropiaEnv* env, const std::string& filename);

int verify(NeuropiaEnv* env);

#endif // NEUROPIA_SIMPLE_H
