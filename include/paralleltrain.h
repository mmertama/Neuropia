#ifndef PARALLELTRAIN_H
#define PARALLELTRAIN_H

#include "trainerbase.h"

namespace Neuropia {

class TrainerParallel : public TrainerBase {
public:
    TrainerParallel(const std::string & root, const Neuropia::Params& params, bool quiet);
    bool train();
protected:
    unsigned jobs;
    size_t batchSize;
};
}



#endif // PARALLELTRAIN_H
