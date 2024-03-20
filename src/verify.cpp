#include "verify.h"
#include "utils.h"
#include <map>

using namespace Neuropia;


std::tuple<int, unsigned> Neuropia::verify(const Neuropia::Layer& network,
                 const std::string& imageFile,
                 const std::string& labelFile,
                 bool quiet,
                 size_t from,
                 size_t count) {
    Neuropia::IdxReader<unsigned char> testImages(imageFile);
    Neuropia::IdxReader<unsigned char> testLabels(labelFile);

    if(!testImages.ok()) {
        std::cerr << "Cannot open images from \"" << imageFile << "\"" << std::endl;
        return std::make_tuple(0, 0);
    }

    if(!testLabels.ok()) {
         std::cerr << "Cannot open labels from \"" << labelFile << "\"" << std::endl;
         return std::make_tuple(0, 0);
    }

    int found = 0;
    Neuropia::timed([&]() {
        const auto imageSize = testImages.size(1) * testImages.size(2);
        std::vector<Neuropia::NeuronType> inputs(imageSize);
        const auto iterations = std::min(count, testLabels.size());
        for(auto i = from; i < iterations; i++) {
            const auto image = testImages.read(imageSize);
            const auto label = static_cast<unsigned>(testLabels.read());

            std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
                return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
            });
            const auto& outputs = network.feed(inputs.begin(), inputs.end());
            const auto max = static_cast<unsigned>(std::distance(outputs.begin(),
                                                 std::max_element(outputs.begin(), outputs.end())));

            if(!quiet) {
                percentage(i, iterations);
            }                                     


#ifdef DEBUG_SHOW
            Neuropia::printimage(image.data()); //ASCII print images
            std::cout << label << "->" << outputs << "->" << max << std::endl; // Print
            std::cout << label << " guessed as " << max << std::endl;
#endif

            if(max == label) {
                ++found;
            }
        }
    }, "Verify");

    return std::make_tuple(found, std::min(count, testLabels.size()) - from);
}

std::tuple<int, unsigned> Neuropia::verifyEnseble(const std::vector<Neuropia::Layer>& ensebles,
                                                  bool hard,
                                                  const std::string& imageFiles,
                                                  const std::string& labelFiles,
                                                  bool quiet,
                                                  size_t from,
                                                  size_t count) {
    Neuropia::IdxReader<unsigned char> testImages(imageFiles);
    Neuropia::IdxReader<unsigned char> testLabels(labelFiles);

    if(!testImages.ok()) {
        std::cerr << "Cannot open images from \"" << imageFiles << "\"" << std::endl;
        return std::make_tuple(0, 0);
    }

    if(!testLabels.ok()) {
         std::cerr << "Cannot open labels from \"" << labelFiles << "\"" << std::endl;
         return std::make_tuple(0, 0);
    }

    const auto imageSize = testImages.size(1) * testImages.size(2);
    int found = 0;
    Neuropia::timed([&]() {
        std::vector<Neuropia::NeuronType> inputs(imageSize);
        const auto iterations = std::min(count, testLabels.size());
        for(auto i = from; i < iterations; i++) {
            const auto image = testImages.read(imageSize);
            const size_t label = testLabels.read();

            std::transform(image.begin(), image.end(), inputs.begin(), [](auto c) {
                return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
            });

            std::map<unsigned, size_t> hardVotes; //hard we take one got most of outputs for each round
            std::map<size_t, NeuronType> softVotes;   //we sum up the results and take one get more over all results in round
            for(const auto& network : ensebles) {
                const auto& outputs = network.feed(inputs.begin(), inputs.end());

                if(hard) {
                    const auto result = static_cast<unsigned>(std::distance(outputs.begin(),
                                                     std::max_element(outputs.begin(), outputs.end())));
                    hardVotes[result]++;
                } else {
                    for(auto it = outputs.begin(); it != outputs.end(); it++) {
                        softVotes[static_cast<unsigned>(std::distance(outputs.begin(), it))] += *it;
                    }
                }
            }

            if(!quiet) {
                percentage(i, iterations);
            }     

            size_t max;

            const auto comp = [](const auto& a, const auto& b){return a.second < b.second;};

            if(hard) {
                max = std::max_element(hardVotes.begin(), hardVotes.end(), comp)->first;
            } else {
                max = static_cast<size_t>(std::distance(softVotes.begin(),
                                                 std::max_element(softVotes.begin(), softVotes.end(), comp)));
            }


             if(max == label) {
                ++found;
            }
        }
    }, "Verify");

    return std::make_tuple(found, std::min(count, testLabels.size()) - from);
}
