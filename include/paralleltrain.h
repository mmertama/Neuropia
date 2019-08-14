#ifndef PARALLELTRAIN_H
#define PARALLELTRAIN_H

#include "trainerbase.h"

namespace Neuropia {

class TrainerParallel : public TrainerBase {
public:
    TrainerParallel(const std::string & root, const Neuropia::Params& params, bool m_quiet);
    bool train() override;
protected:
    unsigned m_jobs;
    size_t m_batchSize;
};
}



#endif // PARALLELTRAIN_H
