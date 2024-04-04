#include <thread>
#include <chrono>
#include "utils.h"
#include "paralleltrain.h"
#include "params.h"

using namespace std::chrono_literals;

using namespace Neuropia;

 TrainerThreaded::TrainerThreaded(const std::string& root, const Neuropia::Params& params, bool quiet) : TrainerBase (root, params, quiet) ,
    m_jobs(params.uinteger("Jobs")),
    m_batchSize(params.uinteger("BatchSize")) {}


 bool TrainerThreaded::callUpdate() {
    const std::lock_guard<std::mutex> lock(m_update_mutex);
    return update();
 }

 bool TrainerThreaded::doInit() {
    m_completed = 0;
    if(!m_network.isValid())
        return false;
    m_offsprings.clear();
    for(auto job = 0U; job < m_jobs; ++job)  {
        m_offsprings.push_back({m_network, std::thread{}}); // copy network for jobs, do no allocate while looping, resize is not working as it copy
    }
    return train();
 }

 void TrainerThreaded::wait() {
    for(auto& o : m_offsprings) { // wait em all
        std::get<std::thread>(o).join();
    }
 }

TrainerThreaded::~TrainerThreaded() {
    for(const auto& o : m_offsprings) { // wait em all
        (void) o;
        neuropia_assert(!std::get<std::thread>(o).joinable());
    }
}  

TrainerParallel::TrainerParallel(const std::string& root, const Neuropia::Params& params, bool quiet)
    : TrainerThreaded (root, params, quiet) {}


bool TrainerParallel::train() {
    const auto inputSize = m_images.size(1) * m_images.size(2);
       for(auto job = 0U; job < m_jobs; ++job)  {
        std::vector<std::tuple<std::vector<unsigned char>, unsigned char>> batchData(m_batchSize);
        for(auto& t : batchData)
            std::get<0>(t).resize(inputSize);
  
        batchData.resize(m_batchSize);

        for(auto i = 0U; i < m_batchSize; i++)  {
            const auto at = m_random.random(m_images.size());
            batchData[i] = {m_images.readAt(at, inputSize), m_labels.readAt(at)};
        }

        // start thread
        std::get<std::thread>(m_offsprings[job]) = std::thread([inputSize, this, batchData = std::move(batchData)](unsigned currentJob) {
            //first we train
            for(auto i = 0U; i < m_batchSize; i++) {

                std::vector<Neuropia::NeuronType> inputs(inputSize);
                std::transform(std::get<0>(batchData[i]).begin(), std::get<0>(batchData[i]).end(), inputs.begin(), [](unsigned char c) {
                    return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
                });

                std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
                outputs[std::get<1>(batchData[i])] = 1.0;

                std::get<Layer>(m_offsprings[currentJob]).train(inputs.begin(), outputs.begin(), this->m_learningRate, this->m_lambdaL2);

                callUpdate();
            }

            ++m_completed;

            // end thread
            }, job);
        }
        return true;
}

bool TrainerParallel::doTrain() {
    if(m_completed < m_jobs) {
        std::this_thread::sleep_for(300ms);
        return true;
    }
    wait();

    for(const auto& o : m_offsprings) {
        m_network.merge( std::get<Layer>(o), 1.0 / static_cast<NeuronType>(m_jobs));
    }

    for(auto& o : m_offsprings) {
        std::get<Layer>(o) = m_network; // copy merged
    }

    return train();
}

bool TrainerParallel::complete() {    
    
    neuropia_assert(m_completed == m_jobs);

    wait();

    //then merge the results by calculate each offspring relative contribution
    for(const auto& o : m_offsprings) {
        m_network.merge( std::get<Layer>(o), 1.0 / static_cast<NeuronType>(m_jobs));
    }

    m_network.inverseDropout();
    return true;
}
