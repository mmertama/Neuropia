#ifndef TRAIN_H
#define TRAIN_H

#include "trainerbase.h"

namespace Neuropia {

/**
 * @brief The Trainer class
 */
class Trainer : public TrainerBase {
public:
   Trainer(const std::string& root, const Neuropia::Params& params, bool m_quiet);
   bool doTrain() override;
};

}

#endif // TRAIN_H
