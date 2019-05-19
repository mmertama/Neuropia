#include "evotrain.h"
#include "params.h"

using namespace Neuropia;

TrainerEvo::TrainerEvo(const std::string& root, const Neuropia::Params& params, bool quiet)
    : TrainerBase(root, params, quiet),
      jobs(params.uinteger("Jobs")),
      batchSize(params.uinteger("BatchSize")),
      batchVerifySize(params.uinteger("BatchVerifySize")){
}

bool TrainerEvo::train() {
    //copy network for jobs
    std::vector<Neuropia::Layer> offsprings(jobs);
    std::fill(offsprings.begin(), offsprings.end(), this->m_network);

    std::vector<std::thread> threads(jobs);
    std::vector<int> results(jobs);
    const auto inputSize = images.size(1) * images.size(2);
    std::vector<std::vector<std::tuple<std::vector<unsigned char>, unsigned char>>> batches(jobs);
    std::vector<std::vector<std::tuple<std::vector<unsigned char>, unsigned char>>> batchesVerify(jobs);

    auto maxNet = 0U;

    int progressCount = 0;
    const auto load = iterations;

    Neuropia::timed([&]() {
        Neuropia::iterator(iterations, [&](size_t it)  {
            ++progressCount;
            for(auto job = 0U; job < jobs; ++job)  {

                auto& batchData = batches[job];
                batchData.resize(batchSize);

                for(auto i = 0U; i < batchSize; i++)  {
                    const auto at = images.random();
                    batchData[i] = {images.next(at, inputSize), labels.next(at)};
                }

                auto& batchVerifyData = batchesVerify[job];
                batchVerifyData.resize(batchVerifySize);

                for(auto i = 0U; i < batchVerifySize; i++)  {
                    const auto at = images.random();
                batchVerifyData[i] = {images.next(at, inputSize), labels.next(at)};
                }

                // start thread
                threads[job] = std::thread([&](unsigned currentJob) {

                    //first we train

                    for(auto i = 0U; i < batchSize; i++) {
                        std::vector<Neuropia::NeuronType> inputData(inputSize);
                        const auto& batch = batchData[i];
                        std::transform(std::get<0>(batch).begin(), std::get<0>(batch).end(), inputData.begin(), [](unsigned char c) {
                            return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                        });

                        std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
                        const auto index = batchData[i];
                        outputs[std::get<1>(index)] = 1.0;

                        offsprings[currentJob].train(inputData.begin(), outputs.begin(), learningRate, lambdaL2);
                    }
                    //then we verify,
                    //it may be debatable to use potentially overlaprogressCounting data for verify batches, but
                    // I assume when samples is small vs. data - and its is on temporary it shall not hard, I definetely wanna
                    // have any risk to contaminate verification data
                    int found = 0;
                    std::vector<Neuropia::NeuronType> inputs(inputSize);
                    for(auto i = 0U; i < batchVerifySize; i++) {

                        std::transform(std::get<0>(batchVerifyData[i]).begin(), std::get<0>(batchVerifyData[i]).end(), inputs.begin(), [](unsigned char c) {
                            return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                        });
                        const auto outputs = offsprings[currentJob].feed(inputs.begin(), inputs.end());
                        const auto max = static_cast<size_t>(std::distance(outputs.begin(),
                                                             std::max_element(outputs.begin(), outputs.end())));
                        if(max == std::get<1>(batchVerifyData[i])) {
                            ++found;
                        }
                    }
                    results[currentJob] = found;
                }, job);

                // end thread
            }

            for(auto& thread : threads) {// wait em all
                thread.join();
            }

            //then find best network
            int maxmax = 0;
            for(auto index = 0U;  index < jobs ; index++) {
                if(results[index] > maxmax) {
                    maxmax = results[index];
                    maxNet = index;
                }
            }
            //copy best of network for jobs
            std::fill(offsprings.begin(), offsprings.end(), offsprings[maxNet]);

            if(maxTrainTime >= MaxTrainTime) {
                if(!quiet)
                    std::cout << "\r" << it << " " << maxmax << " " << results << " "
                              << std::setprecision(3) << (100.0 * (static_cast<double>(progressCount) / static_cast<double>(load))) << '%' << std::flush;
                learningRate += (1.0 / static_cast<double>(iterations)) * (learningRateMin - learningRateMax);
            } else {
                passedIterations = it;
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto delta = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
                if(delta > maxTrainTime) {
                    return false;    //out of time, exit
                }
                const auto change = delta - gap;
                this->gap = delta;
                this->learningRate += (static_cast<double>(change) / static_cast<double>(this->maxTrainTime)) * (this->learningRateMin - this->learningRateMax);

                if(!this->quiet)
                    std::cout << "\r" << it << " " << maxmax << " " << results << " "
                              << std::setprecision(3) << (100.0 * (delta / static_cast<double>(this->maxTrainTime))) << '%' << std::flush;

            }
            return true;
        });
        this->m_network = std::move(offsprings[maxNet]); // then we have done...
        std::cout << std::endl;
    }, "Training evolutionally");
    this->m_network.inverseDropout();
    return true;
    }
