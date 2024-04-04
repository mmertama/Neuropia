#ifndef PARALLELTRAIN_H
#define PARALLELTRAIN_H

#include "trainerbase.h"
#include <atomic>
#include <thread>
#include <mutex>

namespace Neuropia {


class TrainerThreaded : public TrainerBase {
public:
    TrainerThreaded(const std::string & root, const Neuropia::Params& params, bool m_quiet);
    ~TrainerThreaded();
protected:
   bool callUpdate();
   bool doInit() override;
   void wait();
   virtual bool train() = 0;
protected:
    unsigned m_jobs;
    size_t m_batchSize;
    std::vector<std::pair<Neuropia::Layer, std::thread>> m_offsprings{};
    std::atomic<unsigned> m_completed{};
    std::mutex m_update_mutex{};

};

class TrainerParallel : public TrainerThreaded {
public:
    TrainerParallel(const std::string & root, const Neuropia::Params& params, bool m_quiet);
    ~TrainerParallel() = default;
private:
   bool train() override;
   bool complete() override;
   bool doTrain() override;
};
}



#endif // PARALLELTRAIN_H
