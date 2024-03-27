
#include "trainer.h"
#include "verify.h"

using namespace Neuropia;

Trainer::Trainer(const std::string & root, const Neuropia::Params& params, bool quiet) : TrainerBase (root, params, quiet) {
}

bool Trainer::doTrain() {
    #ifdef DO_DUMP_DEBUG
    std::ofstream strm("dump.text", std::ios::app);
#endif

    auto testVerify = m_testVerifyFrequency;

    bool failed = false;
    m_control([&]() {
    Neuropia::iterator(m_iterations, [&](size_t it)->bool {
#ifdef DO_DUMP_DEBUG
        Neuropia::debug(Trainer<inputSize>::network, strm, {1,4});
#endif
        if(m_maxTrainTime >= MaxTrainTime) {
            if(!m_quiet)
                percentage(it + 1, m_iterations);
            m_learningRate += (1.0 / static_cast<NeuronType>(m_iterations)) * (m_learningRateMin - m_learningRateMax);
        } else {
            m_passedIterations = it;
            const auto stop = std::chrono::high_resolution_clock::now();
            const auto delta = static_cast<NeuronType>(std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count());
            if(delta > m_maxTrainTime) {
                return false;
            }
            const auto change = delta - m_gap;
            m_gap = delta;
            if(!m_quiet)
                percentage(delta, m_maxTrainTime, " " + std::to_string(m_learningRate));
            this->m_learningRate += (static_cast<NeuronType>(change) / static_cast<NeuronType>(m_maxTrainTime)) * (m_learningRateMin - m_learningRateMax);
        }

        const auto imageSize = m_images.size(1) * m_images.size(2);
        const auto at = m_random.random(m_images.size());
        const auto image = m_images.readAt(at, imageSize);
        const auto label = static_cast<unsigned>(m_labels.readAt(at));

#ifdef DEBUG_SHOW
            Neuropia::printimage(image.data(), m_images.size(1), m_images.size(2)); //ASCII print images
            std::cout << label << std::endl;
#endif

        std::vector<Neuropia::NeuronType> inputs(imageSize);
        std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
            return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
        });

        std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
        outputs[label] = 1.0; //correct one is 1
        if(!this->m_network.train(inputs.begin(), outputs.begin(), m_learningRate, m_lambdaL2)) {
            failed = true;
            return false;
        }

        if(--testVerify == 0) {
            m_network.inverseDropout();
            printVerify(Neuropia::verify(m_network, m_imageFile, m_labelFile, 0, 200), "Test");
            setDropout();
            testVerify = m_testVerifyFrequency;
        }

        return true;
    });
    std::cout << std::endl;
}, "Training");
m_network.inverseDropout();
#ifdef DO_DUMP_DEBUG
Neuropia::debug(network, strm);
#endif
return !failed;
}
