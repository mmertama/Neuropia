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

extern void testLogicalPorts();

constexpr char topologyRe[] = R"(\d+(,\d+)*$)";
constexpr char activationFunctionRe[] =R"((sigmoid|relu|elu)(,(sigmoid|relu|elu))*$)";
constexpr char dropoutRateRe[] = R"(\d+\.?\d*(,\d+\.?\d*)*$)";


int main(int argc, char* argv[]) {

    Neuropia::Params params = {
        {"ImagesVerify", "", Neuropia::Params::File},
        {"LabelsVerify", "", Neuropia::Params::File},
        {"Images", "", Neuropia::Params::File},
        {"Labels", "", Neuropia::Params::File},
        {"Iterations", "1", Neuropia::Params::Int},
        {"Jobs", "1", Neuropia::Params::Int},
        {"LearningRate", "0", Neuropia::Params::Real},
        {"LearningRateMin", "0.02", Neuropia::Params::Real},
        {"LearningRateMax", "0.02", Neuropia::Params::Real},
        {"BatchSize", "800", Neuropia::Params::Int},
        {"BatchVerifySize", "100", Neuropia::Params::Int},
        {"Topology", "64,32", topologyRe},
        {"MaxTrainTime", std::to_string(MaxTrainTime), Neuropia::Params::Int},
        {"File", "mnistdata.bin", Neuropia::Params::File},
        {"Extra", "", Neuropia::Params::String},
        {"Hard", "false", Neuropia::Params::Bool},
        {"ActivationFunction", "sigmoid", activationFunctionRe},
        {"InitStrategy", "auto", R"((auto|logistic|norm|relu)$)"},
        {"DropoutRate", "0.0", dropoutRateRe},
        {"TestFrequency", "9999999", Neuropia::Params::Int},
        {"L2", "0.0", Neuropia::Params::Real},
        {"Classes", "10", Neuropia::Params::Int}
    };

    bool quiet = false;


    const std::unordered_map<std::string, std::function<void(const std::string&)>> tests = {
    {
            "gates", [](const std::string&) {
                testLogicalPorts();
                std::cout << std::endl;
            }
    },{
            "trainMnist", [&](const std::string & root) {
                Neuropia::Trainer trainer(root, params, quiet);
                const bool ok = trainer.train();
                const auto network = trainer.network();

                if(!params["File"].empty())
                    Neuropia::save(params["File"], network);
                if(ok) {
                    Neuropia::printVerify(verify(network,
                                           Neuropia::absPath(root, params["Images"]),
                                           Neuropia::absPath(root, params["Labels"]), 0, 1000), "Train");
                    Neuropia::printVerify(verify(network,
                                           Neuropia::absPath(root, params["ImagesVerify"]),
                                           Neuropia::absPath(root, params["LabelsVerify"])), "Verify");
                } else {
                    std::cerr << "Training failed ";
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
                const auto ok = trainer.train();
                const auto network = trainer.network();

                if(ok && !params["File"].empty())
                    Neuropia::save(params["File"], network);
                Neuropia::printVerify(verify(network,
                                       Neuropia::absPath(root, params["ImagesVerify"]),
                                       Neuropia::absPath(root, params["LabelsVerify"])), "Verify");
                std::cout << std::endl;
            }
        },
        {
            "trainMnistParallel", [&](const std::string & root) {
               Neuropia::TrainerParallel trainer(root, params, quiet);

               const auto ok = trainer.train();
               const auto network = trainer.network();

                if(ok && !params["File"].empty())
                    Neuropia::save(params["File"], network);
                Neuropia::printVerify(verify(network,
                                       Neuropia::absPath(root, params["ImagesVerify"]),
                                       Neuropia::absPath(root, params["LabelsVerify"])), "Verify");
                std::cout << std::endl;
        }
        },
        {
            "verifyMnist", [&](const std::string & root) {
                Neuropia::Layer network;
                std::ifstream str;
                str.open(params["File"], std::ios::out | std::ios::binary);
                if(str.is_open()) {
                    network = Neuropia::Layer(str);
                } else {
                    std::cerr << params["File"] << " parse error" << std::endl;
                    return;
                }
                str.close();
                Neuropia::printVerify(verify(network, Neuropia::absPath(root, params["ImagesVerify"]),
                         Neuropia::absPath(root, params["LabelsVerify"])), "Verify data");
                std::cout << std::endl;
            },
        },
        {
            "ensemble", [&](const std::string & root){
                const auto files = Neuropia::Params::split(params["Extra"]);
                std::vector<Neuropia::Layer> ensebles;
                for(const auto& file : files) {
                    std::ifstream str;
                    str.open(file, std::ios::out | std::ios::binary);
                    if(str.is_open()) {
                        ensebles.emplace_back(Neuropia::Layer(str));
                    } else {
                        std::cerr << file << " cannot be opened" << std::endl;
                        return;
                    }
                    str.close();
                }
                Neuropia::printVerify(Neuropia::verifyEnseble(ensebles, params.boolean("Hard"),
                        Neuropia::absPath(root, params["ImagesVerify"]),
                        Neuropia::absPath(root, params["LabelsVerify"])), "Verify data");
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
        std::cerr << "neuropia TEST|TESTFILE <test parameters>" << std::endl;
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


