#ifndef VERIFY_H
#define VERIFY_H

#include <algorithm>
#include <memory>
#include "neuropia.h"
#include "idxreader.h"
#include "utils.h"

namespace Neuropia {

    /// @brief Verifier class
    class Verifier {
        public:
        /// @brief D
        ~Verifier();
        /**
         * @brief verify
         * @param network
         * @param imageFile
         * @param labelFile
         * @param from
         * @param count
         * @param random
         */
        Verifier(const Neuropia::Layer& network,
                 const std::string& imageFile,
                 const std::string& labelFile,
                 bool quiet,
                 size_t from = 0, // if random - ignored
                 size_t count = std::numeric_limits<unsigned>::max(),
                 bool random = false);
        /**
         * @brief verify
         * @param network
         * @param imageFile
         * @param labelFile
         * @param from
         * @param count
         * @param random
         */                 
        Verifier(const std::vector<Neuropia::Layer>& ensembles,
                   bool hard,
                   const std::string& imageFile,
                   const std::string& labelFile,
                   bool quiet,
                   size_t from = 0,
                   size_t count = std::numeric_limits<unsigned>::max(),
                   bool random = false);
        /// @brief Iterate one
        /// @return 
        bool next();
        /// @brief get results
        /// @return 
        std::tuple<int, unsigned> result() const;

        /// @brief 
        /// @return 
        NeuronType result_value() const;

        /// @brief timed busy loop and result
        /// @param quiet 
        /// @return 
        std::tuple<int, unsigned> busy(bool quiet = false);

        private:
            class Private;
            std::unique_ptr<Private> m_private;                    
    };
}


#endif // VERIFY_H
