#include "simple.h"
#include "verify.h"


using namespace Neuropia;
using namespace NeuropiaSimple;

bool SimpleTrainer::train(unsigned batchSize, SimpleLogStream& stream) {

    if(m_passedIterations >= m_iterations)
        return false;
    if(m_passedIterations == 0) {
        m_start = std::chrono::high_resolution_clock::now();
    }
    neuropia_assert_always(batchSize > 0, "batchSize must be > 0");
   // stream.freeze(true);
    bool ok = true;
    for(unsigned b = 0; b < std::min(m_iterations, batchSize); ++b) {
        ++m_passedIterations;
        
        const auto success = TrainerBase::train();
    

        if(!success) {
            stream.freeze(false);
            TrainerBase::train(); // to get error out
            m_onEnd(network(), false);
            ok = false;
            break;
        }

        if(m_passedIterations >= m_iterations) {
           stream.freeze(false);
            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << std::endl;
            std::cout << "Training "
                        << "timed:" << std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count()
                        << "." <<  std::chrono::duration_cast<std::chrono::microseconds>(stop - m_start).count()
                        - std::chrono::duration_cast<std::chrono::seconds>(stop - m_start).count() * 1000000 << std::endl;
            m_network.inverseDropout();
            m_onEnd(network(), true);
            break;
            }
        }
        stream.freeze(false);
        return ok;
    }


bool SimpleTrainer::doTrain() {
    auto testVerify = m_testVerifyFrequency;

    if(m_maxTrainTime >= MaxTrainTime) {
        if(!m_quiet)
            percentage(m_passedIterations, m_iterations);
        m_learningRate += (1.0 / static_cast<NeuronType>(m_iterations)) * (m_learningRateMin - m_learningRateMax);
    } else {
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
        return false;
    }

    if(--testVerify == 0) {
        m_network.inverseDropout();
        printVerify(Neuropia::verify(m_network, m_imageFile, m_labelFile, 0, 200), "Test");
        setDropout();
        testVerify = m_testVerifyFrequency;
    }
    return true;
}

bool SimpleVerifier::verify() {
    const auto sz = m_testLabels.size();
    if(m_position >= sz)
        return false;
    const auto imageSize = m_testImages.size(1) * m_testImages.size(2);
    const auto image = m_testImages.read(imageSize);
    const auto label = static_cast<unsigned>(m_testLabels.read());

    std::vector<Neuropia::NeuronType> inputs(imageSize);

    std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
        return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
    });

    ASSERT_X(m_network.isValid(), std::string("On lap" + std::to_string(m_position)).c_str());

    const auto& outputs = m_network.feed(inputs.begin(), inputs.end());
    const auto max = static_cast<unsigned>(std::distance(outputs.begin(),
                                         std::max_element(outputs.begin(), outputs.end())));

    if(max == label) {
        ++m_found;
    }

    ++m_position;
    if((m_position % (sz / 100)) == 0) {
        Neuropia::percentage(m_position, sz);
    }

    return true;
}


