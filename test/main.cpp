#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <map>
#include <cassert>
#include <functional>
#include <tuple>
#include <regex>
#include "neuropia.h"
#include "trainer.h"
#include "paralleltrain.h"
#include "utils.h"
#include "evotrain.h"
#include "verify.h"
#include "params.h"
#include "argparse.h"
#include "default.h"
#include "../neuropialib/neuropialib.h"

extern void testLogicalPorts();

int main(int argc, char* argv[]) {

    Neuropia::Params params = {
        DEFAULT_PARAMS
    };

    bool quiet = false;

    // set testing specific params
    params.set("Classes", "10");
    params.set("File", "mnistdata.bin");

    const std::unordered_map<std::string, std::function<void(const std::string&)>> tests = {
    {
            "gates", [](const std::string&) {
                testLogicalPorts();
                std::cout << std::endl;
            }
    },{
            "trainMnist", [&](const std::string & root) {
                Neuropia::Trainer trainer(root, params, quiet);
                const bool ok = trainer.busy();
                const auto network = trainer.network();

                assert(network.isValid());

                if(!params["File"].empty()) {
                    const auto p = params.toMap();
                    Neuropia::save(params["File"], network, p);
                }
                if(ok) {
                    
                    Neuropia::printVerify(Neuropia::Verifier(network,
                                           Neuropia::absPath(root, params["Images"]),
                                           Neuropia::absPath(root, params["Labels"]), true, 0, 1000).busy(), "Train");
                    Neuropia::printVerify(Neuropia::Verifier(network,
                                           Neuropia::absPath(root, params["ImagesVerify"]),
                                           Neuropia::absPath(root, params["LabelsVerify"]), true).busy(), "Verify");
                } else {
                    std::cerr << "Test training failed ";
                    auto l = &network;
                    while(l) {
                        std::cerr << (l->isValid() ? "valid" : "invalid") << " ";
                        l = l->get(1);
                    }
#ifdef NEUROPIA_DEBUG
                    Neuropia::debug(network);
#endif
                }
                std::cout << std::endl;
            }
        },
        {
            "trainMnistEvo", [&](const std::string & root) {
                Neuropia::TrainerEvo trainer(root, params, quiet);
                const auto ok = trainer.busy();
                const auto network = trainer.network();

                if(ok && !params["File"].empty()) {
                    const auto p = params.toMap();
                    Neuropia::save(params["File"], network);
                }

                Neuropia::printVerify(Neuropia::Verifier(network,
                                       Neuropia::absPath(root, params["ImagesVerify"]),
                                       Neuropia::absPath(root, params["LabelsVerify"]), true).busy(), "Verify");
                std::cout << std::endl;
            }
        },
        {
            "trainMnistParallel", [&](const std::string & root) {
               Neuropia::TrainerParallel trainer(root, params, quiet);

               const auto ok = trainer.busy();
               const auto network = trainer.network();

               if(ok && !params["File"].empty()) {
                   const auto p = params.toMap();
                   Neuropia::save(params["File"], network);
               }

               Neuropia::printVerify(Neuropia::Verifier(network,
                                       Neuropia::absPath(root, params["ImagesVerify"]),
                                       Neuropia::absPath(root, params["LabelsVerify"]), true).busy(), "Verify");
               std::cout << std::endl;
            }
        },
        {
            "verifyMnist", [&](const std::string & root) {

                const auto loaded = Neuropia::load(params["File"]);
                if(!loaded) {
                    std::cerr << params["File"] << " parse error" << std::endl;
                    return;
                }
                for(const auto& p : std::get<1>(*loaded))
                    std::cout << p.first << "=" << p.second << std::endl;
                std::cout << std::endl;

               

                Neuropia::printVerify(Neuropia::Verifier(std::get<0>(*loaded), Neuropia::absPath(root, params["ImagesVerify"]),
                         Neuropia::absPath(root, params["LabelsVerify"]), true).busy(), "Verify data");
                std::cout << std::endl;
            },
        },
        {
            "ensemble", [&](const std::string & root){
                const auto files = Neuropia::Params::split(params["Extra"]);
                std::vector<Neuropia::Layer> ensembles;
                for(const auto& file : files) {
                    const auto loaded = Neuropia::load(file);
                    if(loaded) {
                        ensembles.emplace_back(std::get<0>(*loaded));
                    } else {
                        std::cerr << file << " cannot be opened" << std::endl;
                        return;
                    }
                }
                Neuropia::printVerify(Neuropia::Verifier(ensembles, params.boolean("Hard"),
                        Neuropia::absPath(root, params["ImagesVerify"]),
                        Neuropia::absPath(root, params["LabelsVerify"]), true).busy(), "Verify data");
                std::cout << std::endl;
            }
        }
    };

    ArgParse argparse;
    argparse
            .addOpt('v', "verbose")
            .addOpt('q', "quiet")
            .addOpt('r', "root", true)
            .addOpt('s', "save", true, "mnistdata.bin");

    if(!argparse.set(argc, argv)) {
        std::cerr << "Invalid args" << std::endl;
        return -1;
    }

    if(argparse.paramCount() > 1 && tests.find(argv[1]) == tests.end()) {
        quiet = true;

        std::string root = ".";
        if(argparse.hasOption('r')) {
            root = argparse.option('r');
            }
            if(argparse.hasOption('v')) {
                quiet = false;
            }
            if(argparse.hasOption('q')) {
                quiet = true;
            }

        if(!params.readTask(argparse.param(1), tests, root)) {
            std::cerr << "file " << argparse.param(1) << " cannot be open" << std::endl;
        }
    } else if(argparse.paramCount() == 1 || tests.find(argv[1]) == tests.end()) {
        std::cerr << "neuropia_test TEST|TESTFILE <test parameters>" << std::endl;
        std::cerr << "e.g trainMnistEvo neuropia/data/t10k-images-idx3-ubyte neuropia/data/t10k-labels-idx1-ubyte neuropia/data/train-images-idx3-ubyte neuropia/data/train-labels-idx1-ubyte 10 8" << std::endl;
        std::cerr << "e.g tests2.txt -r data/mnist" << std::endl;
        std::cerr << "tests:" << std::endl;
        for(const auto& p :  tests) {
            std::cerr << p.first << std::endl;
        }
        std::cerr << "hyperparameters:" << std::endl;
        params.addHelp( topologyRe, "\',\'-separated list of integers");
        params.addHelp( activationFunctionRe, "\',\'-separated list of activation functions: \"sigmoid, relu or elu\"");
        params.addHelp( dropoutRateRe, "\',\'-separated of list of real numbers");
        for(const auto& p :  params) {
            std::cerr  << std::get<0>(p) <<  ", default:\"" << std::get<1>(p) << "\" assuming: \"" << params.toType(std::get<2>(p)) << "\"" << std::endl;
        }
    } else {
        params.readParams(argparse);
        tests.at(argparse.param(1))(".");
    }

    return 0;
}


