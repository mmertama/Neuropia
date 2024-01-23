#include <thread>
#include "utils.h"
#include "paralleltrain.h"
#include "params.h"

using namespace Neuropia;

TrainerParallel::TrainerParallel(const std::string& root, const Neuropia::Params& params, bool quiet)
    : TrainerBase (root, params, quiet),
    m_jobs(params.uinteger("Jobs")),
    m_batchSize(params.uinteger("BatchSize")){
}


bool TrainerParallel::train() {
//copy network for jobs
    std::vector<Neuropia::Layer> offsprings(m_jobs);
    std::vector<std::thread> threads(m_jobs);
    std::vector<std::vector<std::tuple<std::vector<unsigned char>, unsigned char>>> batches(m_jobs);
    for(auto&v : batches)
        for(auto& t : v)
        std::get<0>(t).resize(m_images.size(1) * m_images.size(2));

    int progressCount = 0;
    const auto load = m_jobs * this->m_iterations;

Neuropia::timed([&]() {
    Neuropia::iterator(this->m_iterations, [&](size_t it)  {
        ++progressCount;

        if(m_maxTrainTime >= MaxTrainTime) {
            if(!this->m_quiet)
                std::cout << "\r"
                          << std::setprecision(2) << (100.0 * (static_cast<NeuronType>(progressCount) / static_cast<NeuronType>(load))) << '%' << std::flush;
        } else {
            this->m_passedIterations = it;
            const auto stop = std::chrono::high_resolution_clock::now();
            const auto delta = static_cast<NeuronType>(std::chrono::duration_cast<std::chrono::seconds>(stop - this->m_start).count());
            if(delta > this->m_maxTrainTime) {
                return false;    //out of time, exit
            }
            const auto change = delta - this->m_gap;
            this->m_gap = delta;
            this->m_learningRate += (static_cast<NeuronType>(change) / static_cast<NeuronType>(this->m_maxTrainTime)) * (this->m_learningRateMin - this->m_learningRateMax);

            if(!this->m_quiet)
                std::cout << "\r"
                          << std::setprecision(3) << (100.0 * (delta / static_cast<NeuronType>(this->m_maxTrainTime))) << '%' << std::flush;
        }

        //copy network for jobs
        std::fill(offsprings.begin(), offsprings.end(), this->m_network);
        const auto inputSize = m_images.size(1) * m_images.size(2);

        for(auto job = 0U; job < m_jobs; ++job)  {

            auto& batchData = batches[job];
            batchData.resize(m_batchSize);

            for(auto i = 0U; i < m_batchSize; i++)  {
                const auto at = this->m_images.random();
                batchData[i] = {this->m_images.next(at, inputSize), this->m_labels.next(at)};
            }

            // start thread
            threads[job] = std::thread([&](unsigned currentJob) {
                //first we train
                for(auto i = 0U; i < m_batchSize; i++) {

                    std::vector<Neuropia::NeuronType> inputs(inputSize);
                    std::transform(std::get<0>(batchData[i]).begin(), std::get<0>(batchData[i]).end(), inputs.begin(), [](unsigned char c) {
                        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                    });

                    std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
                    outputs[std::get<1>(batchData[i])] = 1.0;

                    offsprings[currentJob].train(inputs.begin(), outputs.begin(), this->m_learningRate, this->m_lambdaL2);
                }
             // end thread
            }, job);
        }
        for(auto& thread : threads) { // wait em all
            thread.join();
        }

        //then merge the results by caclucate each offspring relative contribution
        for(const auto& offspring : offsprings) {
            this->m_network.merge(offspring, 1.0 / static_cast<NeuronType>(m_jobs));
        }

        return true;
    });

    std::cout << std::endl;
}, "Training contributionally");
this->m_network.inverseDropout();
return true;
}
