#ifndef VERIFY_H
#define VERIFY_H

#include <algorithm>
#include "neuropia.h"
#include "idxreader.h"
#include "utils.h"

namespace Neuropia {
/**
 * @brief verify
 * @param network
 * @param imageFile
 * @param labelFile
 * @param from
 * @param count
 * @return values found, values looked 
 */
std::tuple<int, unsigned> verify(const Neuropia::Layer& network,
                 const std::string& imageFile,
                 const std::string& labelFile,
                 bool quiet,
                 size_t from = 0,
                 size_t count = std::numeric_limits<unsigned>::max());
/**
 * @brief verifyEnseble
 * @param ensebles
 * @param hard
 * @param imageFile
 * @param labelFile
 */
std::tuple<int, unsigned> verifyEnseble(const std::vector<Neuropia::Layer>& ensebles,
                   bool hard,
                   const std::string& imageFile,
                   const std::string& labelFile,
                   bool quiet,
                   size_t from = 0,
                   size_t count = std::numeric_limits<unsigned>::max());
}


#endif // VERIFY_H
