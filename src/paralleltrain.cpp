#include <thread>
#include "utils.h"
#include "paralleltrain.h"
#include "params.h"

using namespace Neuropia;

TrainerParallel::TrainerParallel(const std::string& root, const Neuropia::Params& params, bool quiet)
    : TrainerBase (root, params, quiet),
    jobs(params.uinteger("Jobs")),
    batchSize(params.uinteger("BatchSize")){
}


bool TrainerParallel::train() {
//copy network for jobs
    std::vector<Neuropia::Layer> offsprings(jobs);
    std::vector<std::thread> threads(jobs);
    std::vector<std::vector<std::tuple<std::vector<unsigned char>, unsigned char>>> batches(jobs);
    for(auto&v : batches)
        for(auto& t : v)
        std::get<0>(t).resize(images.size(1) * images.size(2));

    int progressCount = 0;
    const auto load = jobs * this->iterations;

Neuropia::timed([&]() {
    Neuropia::iterator(this->iterations, [&](size_t it)  {
        ++progressCount;

        if(maxTrainTime >= MaxTrainTime) {
            if(!this->quiet)
                std::cout << "\r"
                          << std::setprecision(2) << (100.0 * (static_cast<double>(progressCount) / static_cast<double>(load))) << '%' << std::flush;
        } else {
            this->passedIterations = it;
            const auto stop = std::chrono::high_resolution_clock::now();
            const auto delta = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(stop - this->start).count());
            if(delta > this->maxTrainTime) {
                return false;    //out of time, exit
            }
            const auto change = delta - this->gap;
            this->gap = delta;
            this->learningRate += (static_cast<double>(change) / static_cast<double>(this->maxTrainTime)) * (this->learningRateMin - this->learningRateMax);

            if(!this->quiet)
                std::cout << "\r"
                          << std::setprecision(3) << (100.0 * (delta / static_cast<double>(this->maxTrainTime))) << '%' << std::flush;
        }

        //copy network for jobs
        std::fill(offsprings.begin(), offsprings.end(), this->m_network);
        const auto inputSize = images.size(1) * images.size(2);

        for(auto job = 0U; job < jobs; ++job)  {

            auto& batchData = batches[job];
            batchData.resize(batchSize);

            for(auto i = 0U; i < batchSize; i++)  {
                const auto at = this->images.random();
                batchData[i] = {this->images.next(at, inputSize), this->labels.next(at)};
            }

            // start thread
            threads[job] = std::thread([&](unsigned currentJob) {
                //first we train
                for(auto i = 0U; i < batchSize; i++) {

                    std::vector<Neuropia::NeuronType> inputs(inputSize);
                    std::transform(std::get<0>(batchData[i]).begin(), std::get<0>(batchData[i]).end(), inputs.begin(), [](unsigned char c) {
                        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                    });

                    std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
                    outputs[std::get<1>(batchData[i])] = 1.0;

                    offsprings[currentJob].train(inputs.begin(), outputs.begin(), this->learningRate, this->lambdaL2);
                }
             // end thread
            }, job);
        }
        for(auto& thread : threads) { // wait em all
            thread.join();
        }

        //then merge the results by caclucate each offspring relative contribution
        for(const auto& offspring : offsprings) {
            this->m_network.merge(offspring, 1.0 / static_cast<double>(jobs));
        }

        return true;
    });

    std::cout << std::endl;
}, "Training contributionally");
this->m_network.inverseDropout();
return true;
}
