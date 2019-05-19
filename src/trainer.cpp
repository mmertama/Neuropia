
#include "trainer.h"
#include "verify.h"

using namespace Neuropia;

Trainer::Trainer(const std::string & root, const Neuropia::Params& params, bool quiet) : TrainerBase (root, params, quiet) {
}

bool Trainer::train() {
    #ifdef DO_DUMP_DEBUG
    std::ofstream strm("dump.text", std::ios::app);
#endif
    auto testVerify = testVerifyFrequency;

    bool failed = false;
control([&]() {
    Neuropia::iterator(iterations, [&](size_t it)->bool {
#ifdef DO_DUMP_DEBUG
        Neuropia::debug(Trainer<inputSize>::network, strm, {1,4});
#endif
        if(maxTrainTime >= MaxTrainTime) {
            if(!quiet)
                persentage(it + 1, iterations);
            learningRate += (1.0 / static_cast<double>(iterations)) * (learningRateMin - learningRateMax);
        } else {
            passedIterations = it;
            const auto stop = std::chrono::high_resolution_clock::now();
            const auto delta = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
            if(delta > maxTrainTime) {
                return false;
            }
            const auto change = delta - gap;
            gap = delta;
            if(!quiet)
                persentage(delta, maxTrainTime, " " + std::to_string(learningRate));
            this->learningRate += (static_cast<double>(change) / static_cast<double>(maxTrainTime)) * (learningRateMin - learningRateMax);
        }

        const auto imageSize = images.size(1) * images.size(2);
        const auto at = images.random();
        const auto image = images.next(at, imageSize);
        const auto label = static_cast<unsigned>(labels.next(at));

#ifdef DEBUG_SHOW
            Neuropia::printimage(image.data()); //ASCII print images
            std::cout << label << std::endl;
#endif

        std::vector<Neuropia::NeuronType> inputs(imageSize);
        std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
            return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
        });

        std::vector<Neuropia::NeuronType> outputs(m_network.outLayer()->size());
        outputs[label] = 1.0; //correct one is 1
        if(!this->m_network.train(inputs.begin(), outputs.begin(), learningRate, lambdaL2)) {
            failed = true;
            return false;
        }

        if(--testVerify == 0) {
            m_network.inverseDropout();
            printVerify(Neuropia::verify(m_network, imageFile, labelFile, 0, 200), "Test");
            setDropout();
            testVerify = testVerifyFrequency;
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
