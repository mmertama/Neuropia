#include "evotrain.h"
#include "params.h"

using namespace Neuropia;

TrainerEvo::TrainerEvo(const std::string& root, const Neuropia::Params& params, bool quiet)
    : TrainerBase(root, params, quiet),
      m_jobs(params.uinteger("Jobs")),
      m_batchSize(params.uinteger("BatchSize")),
      m_batchVerifySize(params.uinteger("BatchVerifySize")){
}

bool TrainerEvo::train() {
    //copy network for jobs
    std::vector<Neuropia::Layer> offsprings(m_jobs);
    std::fill(offsprings.begin(), offsprings.end(), this->m_network);

    std::vector<std::thread> threads(m_jobs);
    std::vector<int> results(m_jobs);
    const auto inputSize = m_images.size(1) * m_images.size(2);
    std::vector<std::vector<std::tuple<std::vector<unsigned char>, unsigned char>>> batches(m_jobs);
    std::vector<std::vector<std::tuple<std::vector<unsigned char>, unsigned char>>> batchesVerify(m_jobs);

    auto maxNet = 0U;

    int progressCount = 0;
    const auto load = m_iterations;

    Neuropia::timed([&]() {
        Neuropia::iterator(m_iterations, [&](size_t it)  {
            ++progressCount;
            for(auto job = 0U; job < m_jobs; ++job)  {

                auto& batchData = batches[job];
                batchData.resize(m_batchSize);

                for(auto i = 0U; i < m_batchSize; i++)  {
                    const auto at = m_images.random();
                    batchData[i] = {m_images.next(at, inputSize), m_labels.next(at)};
                }

                auto& batchVerifyData = batchesVerify[job];
                batchVerifyData.resize(m_batchVerifySize);

                for(auto i = 0U; i < m_batchVerifySize; i++)  {
                    const auto at = m_images.random();
                batchVerifyData[i] = {m_images.next(at, inputSize), m_labels.next(at)};
                }

                // start thread
                threads[job] = std::thread([&](unsigned currentJob) {

                    //first we train

                    for(auto i = 0U; i < m_batchSize; i++) {
                        std::vector<Neuropia::NeuronType> inputData(inputSize);
                        const auto& batch = batchData[i];
                        std::transform(std::get<0>(batch).begin(), std::get<0>(batch).end(), inputData.begin(), [](unsigned char c) {
                            return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                        });

                        std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
                        const auto index = batchData[i];
                        outputs[std::get<1>(index)] = 1.0;

                        offsprings[currentJob].train(inputData.begin(), outputs.begin(), m_learningRate, m_lambdaL2);
                    }
                    //then we verify,
                    //it may be debatable to use potentially overlaprogressCounting data for verify batches, but
                    // I assume when samples is small vs. data - and its is on temporary it shall not hard, I definetely wanna
                    // have any risk to contaminate verification data
                    int found = 0;
                    std::vector<Neuropia::NeuronType> inputs(inputSize);
                    for(auto i = 0U; i < m_batchVerifySize; i++) {

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
            for(auto index = 0U;  index < m_jobs ; index++) {
                if(results[index] > maxmax) {
                    maxmax = results[index];
                    maxNet = index;
                }
            }
            //copy best of network for jobs
            std::fill(offsprings.begin(), offsprings.end(), offsprings[maxNet]);

            if(m_maxTrainTime >= MaxTrainTime) {
                if(!m_quiet)
                    std::cout << "\r" << it << " " << maxmax << " " << results << " "
                              << std::setprecision(3) << (100.0 * (static_cast<double>(progressCount) / static_cast<double>(load))) << '%' << std::flush;
                m_learningRate += (1.0 / static_cast<double>(m_iterations)) * (m_learningRateMin - m_learningRateMax);
            } else {
                m_passedIterations = it;
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto delta = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count());
                if(delta > m_maxTrainTime) {
                    return false;    //out of time, exit
                }
                const auto change = delta - m_gap;
                this->m_gap = delta;
                this->m_learningRate += (static_cast<double>(change) / static_cast<double>(this->m_maxTrainTime)) * (this->m_learningRateMin - this->m_learningRateMax);

                if(!this->m_quiet)
                    std::cout << "\r" << it << " " << maxmax << " " << results << " "
                              << std::setprecision(3) << (100.0 * (delta / static_cast<double>(this->m_maxTrainTime))) << '%' << std::flush;

            }
            return true;
        });
        this->m_network = std::move(offsprings[maxNet]); // then we have done...
        std::cout << std::endl;
    }, "Training evolutionally");
    this->m_network.inverseDropout();
    return true;
    }
