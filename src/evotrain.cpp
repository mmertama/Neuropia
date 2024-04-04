#include <chrono>
#include "evotrain.h"
#include "params.h"
#include "verify.h"

using namespace std::chrono_literals;

using namespace Neuropia;

TrainerEvo::TrainerEvo(const std::string& root, const Neuropia::Params& params, bool quiet)
    : TrainerThreaded(root, params, quiet),
      m_batchVerifySize(params.uinteger("BatchVerifySize")) {
}

bool TrainerEvo::train() {
    const auto inputSize = m_images.size(1) * m_images.size(2);         
    for(auto job = 0U; job < m_jobs; ++job)  {
        std::vector<std::tuple<std::vector<unsigned char>, unsigned char>> batchData(m_batchSize);
        for(auto& t : batchData)
            std::get<0>(t).resize(inputSize);
  
        batchData.resize(m_batchSize);

        m_completed = 0;
        const auto iterations = this->iterations();
        std::get<std::thread>(m_offsprings[job]) = std::move(std::thread([iterations, inputSize, this, batchData = std::move(batchData)](unsigned currentJob) {

            for(auto i = 0U; i < m_batchSize; i++) {
                std::vector<Neuropia::NeuronType> inputData(inputSize);
                const auto& batch = batchData[i];
                std::transform(std::get<0>(batch).begin(), std::get<0>(batch).end(), inputData.begin(), [](unsigned char c) {
                    return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                });

                std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
                const auto index = batchData[i];
                outputs[std::get<1>(index)] = 1.0;

                std::get<Layer>(m_offsprings[currentJob]).train(inputData.begin(), outputs.begin(), m_learningRate, m_lambdaL2);
                callUpdate();
            }
            //then we verify,
            //it may be debatable to use potentially overlaprogressCounting data for verify batches, but
            // I assume when samples is small vs. data - and its is on temporary it shall not hard, I definitely wanna
            // have any risk to contaminate verification data
            const auto result = Neuropia::Verifier(std::get<Layer>(m_offsprings[currentJob]), m_imageFile, m_labelFile, true, 0, m_batchVerifySize, true).busy(true);
            m_results[currentJob] = std::get<0>(result);
            return true;
        }, job));
    }
    return true;
}

bool TrainerEvo::doTrain() {
    if(m_completed < m_jobs) {
        std::this_thread::sleep_for(300ms);
        return true;
    }

    wait();

    //then find best network
    int maxmax = 0;
    for(auto index = 0U;  index < m_jobs ; index++) {
        if(m_results[index] > maxmax) {
            maxmax = m_results[index];
            m_maxNet = index;
        }
    }

    //copy best of network for jobs
    for(auto& o : m_offsprings) {
        std::get<Layer>(o) =  std::get<Layer>(m_offsprings[m_maxNet]);
    }
    return train();
}

         
bool TrainerEvo::complete() {  
    neuropia_assert(m_completed == m_jobs);

    wait();
      
    m_network = std::get<Layer>(m_offsprings[m_maxNet]); // then we have done...
    std::cout << std::endl;
    m_network.inverseDropout();
    return true;
    }
