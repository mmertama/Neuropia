#ifndef EVOTRAIN_H
#define EVOTRAIN_H

#include <vector>
#include <algorithm>
#include <thread>
#include <iomanip>
#include <array>
#include <atomic>
#include "neuropia.h"
#include "idxreader.h"
#include "utils.h"
#include "paralleltrain.h"

namespace Neuropia {

class TrainerEvo : public TrainerThreaded {
public:
    TrainerEvo(const std::string & root, const Neuropia::Params& params, bool quiet);
    ~TrainerEvo() = default;
private:
   bool complete() override;
   bool doTrain() override;
   bool train() override;
protected:
    unsigned m_batchVerifySize;
    unsigned m_maxNet = 0;
    std::vector<int> m_results = {};
};

}

#endif // EVOTRAIN_H
