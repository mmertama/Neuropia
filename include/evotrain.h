#ifndef EVOTRAIN_H
#define EVOTRAIN_H

#include <vector>
#include <algorithm>
#include <thread>
#include <iomanip>
#include <array>
#include "neuropia.h"
#include "idxreader.h"
#include "utils.h"
#include "trainerbase.h"

namespace Neuropia {

class TrainerEvo : public TrainerBase {
public:
    TrainerEvo(const std::string & root, const Neuropia::Params& params, bool m_quiet);
    bool train();
protected:
    unsigned m_jobs;
    size_t m_batchSize;
    size_t m_batchVerifySize;
};

}

#endif // EVOTRAIN_H
