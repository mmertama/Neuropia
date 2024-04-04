
#include "trainer.h"


using namespace Neuropia;

Trainer::Trainer(const std::string & root, const Neuropia::Params& params, bool quiet) : TrainerBase (root, params, quiet) {
}

bool Trainer::doInit() {
#ifdef DO_DUMP_DEBUG
    std::ofstream strm("dump.text", std::ios::app);
    Neuropia::debug(Trainer<inputSize>::network, strm, {1,4});
#endif
    return true;
}

bool Trainer::doTrain() {

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

        if(!verify())
            return false;

        update();
        return true;
    }

bool Trainer::complete() {
    m_network.inverseDropout();
    #ifdef DO_DUMP_DEBUG
    Neuropia::debug(network, strm);
    #endif
    return true;
}
