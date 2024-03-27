#include "neuropia_feed.h"
#include "neuropia_bin.h"
#include "idxreader.h"
#include "utils.h"
#include "argparse.h"
#include <iostream>
#include <filesystem>

using NeuropiaFeed = Neuropia::Feed<neuropia_bin, sizeof(neuropia_bin)>;

int main(int argc, char* argv[]) {

    static_assert(NeuropiaFeed::in_layer_size() == 28 * 28); // MNIST data size
    static_assert(NeuropiaFeed::out_layer_size() ==  10);    // gives you digits!
    ArgParse argparse;
    argparse.addOpt('q', "quiet");
    argparse.set(argc, argv);
    const auto is_quiet = argparse.hasOption("quiet");
    
    if(!is_quiet) {
        std::cout << "bytes: " << sizeof(neuropia_bin) << 
        "\nlayers: " << static_cast<int>(NeuropiaFeed::layer_count()) << 
        "\nparameters: " << static_cast<int>(NeuropiaFeed::parameter_count()) << std::endl;
        for(auto i = 0U; i < NeuropiaFeed::parameter_count(); ++i) {
            const auto& [key, value] = NeuropiaFeed::parameter(i) ;
            std::cout << i + 1 << ") " << key << "->" << value << std::endl;
        }

        for(auto i = 0U; i < NeuropiaFeed::layer_count(); ++i) {
            const auto& [sz, af] = NeuropiaFeed::layer_info(i);
            std::cout << "L:" << i << " " << sz << " " << af  << std::endl;
        }
    }

    if(argc < 3) {
        std::cerr << "Expect IMAGE_IDX LABELS_IDX" << std::endl;
        return 1;
    }

    Neuropia::IdxReader<unsigned char> testImages(argv[1]);
    Neuropia::IdxReader<unsigned char> testLabels(argv[2]);


    if(!testImages.ok()) {
        std::cerr << "Bad IMAGE_IDX " << argv[1] << std::endl;
        return 1;
    }

     if(!testLabels.ok()) {
        std::cerr << " Bad LABELS_IDX " << argv[2] << std::endl;
        return 1;
    }

    int found = 0;
    Neuropia::timed([&]() {
        const auto imageSize = testImages.size(1) * testImages.size(2);
        std::vector<Neuropia::NeuronType> inputs(imageSize);
        const auto iterations = testLabels.size();
        for(auto i = 0U; i < iterations; i++) {
            const auto image = testImages.read(imageSize);
            const auto label = static_cast<unsigned>(testLabels.read());

            std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
                return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
            });
            const auto outputs = NeuropiaFeed::feed(inputs.begin(), inputs.end());
            const auto max = static_cast<unsigned>(std::distance(outputs.begin(),
                                                 std::max_element(outputs.begin(), outputs.end())));

            Neuropia::percentage(i, iterations);                                     

#ifdef DEBUG_SHOW
            Neuropia::printimage(image.data(), testImages.size(1), testImages.size(2)); //ASCII print images
            std::cout << label << "->" << outputs << "->" << max << std::endl; // Print
            std::cout << label << " guessed as " << max << std::endl;
#endif

            if(max == label) {
                ++found;
            }
        }
    }, "Verify");

    std::cout << "verify index:" << (static_cast<double>(found) * 100.)/ static_cast<double>(testLabels.size()) << " size:" << testLabels.size() << std::endl; 

    return 0;
}


