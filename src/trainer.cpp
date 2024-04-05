
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
        neuropia_assert_always(at < m_labels.size(), "Too big!");
        const auto label = static_cast<unsigned>(m_labels.readAt(at));

#ifdef DEBUG_SHOW
            Neuropia::printimage(image.data(), m_images.size(1), m_images.size(2)); //ASCII print images
            std::cout << label << std::endl;
#endif

        neuropia_assert_always(imageSize > 0, "Invalid image");
        std::vector<Neuropia::NeuronType> inputs(imageSize);
        std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
            return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
        });

        neuropia_assert_always(m_network.outLayer()->size() > 0, "Invalid classes");
        std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());

        if(label >= outputs.size()) {
            std::cerr << "\nError due data mismatch! The labels refers to value at " << label << " when size is " << outputs.size() << std::endl;
            std::cerr << "Images size: " <<  m_images.size() << " Labels size:" <<  m_labels.size() << " a label " << at << std::endl;
            if(!m_labels.verify<unsigned char>(0, m_labels.size(), 0, static_cast<unsigned char>(classes()))) {
                std::cerr << "Bad labels data in " <<  m_labelFile << std::endl;
            }

            return false;
        }

        outputs[label] = static_cast<Neuropia::NeuronType>(1.0); // correct one is 1
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
