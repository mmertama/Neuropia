#include "argparse.h"
#include "neuropia_simple.h"
#include <iostream>


int main(int argc, char* argv[]) {

    ArgParse argparse;
    argparse.addOpt('d', "data_type", true, "double");

    if(!argparse.set(argc, argv)) {
        std::cerr << "Invalid args" << std::endl;
        return -1;
    }

    Neuropia::SaveType save_type = Neuropia::SaveType::SameAsNeuronType;

    if(argparse.hasOption("data_type")) {
        if(argparse.option("data_type") == "double")
            save_type = Neuropia::SaveType::Double;
        else if(argparse.option("data_type") == "float")
            save_type = Neuropia::SaveType::Float;
        else if(argparse.option("data_type") == "long")
            save_type = Neuropia::SaveType::LongDouble;
        else if(argparse.option("data_type") == "longDouble")
            save_type = Neuropia::SaveType::LongDouble;
        else {
            std::cerr << "Bad data type --data_type <double|float|longDouble>";
        }                        
    }

    auto neuropia = NeuropiaSimple::create("");
   
    if(argparse.paramCount() < 4) {
        std::cerr << "neuropia <--data_type <double|float|longDouble>> DATA LABELS OUTPUT <PARAMS>" << std::endl;
        std::cerr << "Where params is KEY=VALUE:" << std::endl;
        std::cerr << "data_type option defines if neurons are stored in float (32 bit), double (64 bit) or long double (128 bit) precision. Default is a build option, and it defaults to double." << std::endl;
        for(const auto& [k, v] :  NeuropiaSimple::params(neuropia)) {
            std::cerr << "'"<< k << "', as " << v.front() << std::endl;
        }
        return 1;
    }


    if(!NeuropiaSimple::setParam(neuropia, "Images", argparse.param(1))) {
        std::cerr << "Bad parameter DATA" << std::endl;
        return 1;
    }

    if(!NeuropiaSimple::setParam(neuropia, "Labels", argparse.param(2))) {
        std::cerr << "Bad parameter LABELS" << std::endl;
        return 1;
    }

    if(!NeuropiaSimple::setParam(neuropia, "File", argparse.param(3))) {
        std::cerr << "Bad parameter OUTPUT" << std::endl;
        return 1;
    }

    for(auto i = 4U; i < argparse.paramCount(); ++i) {
        const auto param = argparse.param(i);
        const auto eq = param.find('=');
        if(eq == std::string::npos || eq == 0) {
            std::cerr << "Bad parameter '"<< param <<"', expects KEY=VALUE" << std::endl;
            return 1;
        }
        const auto key = param.substr(0, eq);
        const auto value = param.substr(eq + 1);
        if(!NeuropiaSimple::setParam(neuropia, key, value)) {
            std::cerr << "Bad optional parameter '" << key << "' as '" << value << "'" << std::endl;
            return 1;
        }
    }

    if(!NeuropiaSimple::train(neuropia, NeuropiaSimple::TrainType::Basic)) {
         std::cerr << "Train failed" << std::endl;
            return 2;
    }

    const auto params = NeuropiaSimple::params(neuropia);
    const auto verify_images = params.find("ImagesVerify");
    const auto verify_labels = params.find("LabelsVerify");
    if(verify_labels != params.end() && verify_images != params.end()) {
        const size_t vers = 10000;   
        const auto result = NeuropiaSimple::verify(neuropia, vers);
        if(std::get<0>(result) < (vers / 3U)) {
            std::cerr << "Supposedly network training was not successful " << std::get<1>(result) * 100 << "%" << std::endl;
            return 3;
        }
    }

    NeuropiaSimple::save(neuropia, argparse.param(3), save_type);

    return 0;
}


