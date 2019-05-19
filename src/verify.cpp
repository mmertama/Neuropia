#include <map>
#include "verify.h"

using namespace Neuropia;

std::tuple<int, unsigned> Neuropia::verify(const Neuropia::Layer& network,
                 const std::string& imageFile,
                 const std::string& labelFile,
                 size_t from,
                 size_t count) {
    Neuropia::IdxReader<unsigned char> testImages(imageFile);
    neuropia_assert_always(testImages.ok(), imageFile);
    Neuropia::IdxReader<unsigned char> testLabels(labelFile);
    neuropia_assert_always(testLabels.ok(), labelFile);

    int found = 0;
    Neuropia::timed([&]() {
        const auto imageSize = testImages.size(1) * testImages.size(2);
        std::vector<Neuropia::NeuronType> inputs(imageSize);
        for(auto i = from; i < std::min(count, testLabels.size()); i++) {
            const auto image = testImages.next(imageSize);
            const auto label = static_cast<unsigned>(testLabels.next());

            std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
                return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
            });
            const auto& outputs = network.feed(inputs.begin(), inputs.end());
            const auto max = static_cast<unsigned>(std::distance(outputs.begin(),
                                                 std::max_element(outputs.begin(), outputs.end())));


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

void Neuropia::verifyEnseble(const std::vector<Neuropia::Layer>& ensebles, bool hard, const std::string& imageFiles, const std::string& labelFiles) {
    Neuropia::IdxReader<unsigned char> testImages(imageFiles);
    neuropia_assert_always(testImages.ok(), imageFiles);
    Neuropia::IdxReader<unsigned char> testLabels(labelFiles);
    neuropia_assert_always(testLabels.ok(), labelFiles);


    const auto imageSize = testImages.size(1) * testImages.size(2);
    int found = 0;
    Neuropia::timed([&]() {
        std::vector<Neuropia::NeuronType> inputs(imageSize);
        for(size_t i = 0; i < testLabels.size(); i++) {
            const auto image = testImages.next(imageSize);
            const size_t label = testLabels.next();

            std::transform(image.begin(), image.end(), inputs.begin(), [](auto c) {
                return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
            });

            std::map<unsigned, size_t> hardVotes; //hard we take one got most of outputs for each round
            std::map<size_t, double> softVotes;   //we sum up the results and take one get more over all results in round
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

    std::cout << std::endl;
    std::cout << "Success fate:" << 100.0 * (static_cast<double>(found) / static_cast<double>(testLabels.size()))
              << "%, found:" << found
              << " of " << testLabels.size() << std::endl;
}
