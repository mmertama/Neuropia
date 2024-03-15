#include "argparse.h"
#include "neuropia_simple.h"
#include "utils.h"
#include <iostream>


int main(int argc, char* argv[]) {

    ArgParse argparse;
    argparse.addOpt('d', "data_type", true, "double");

    if(!argparse.set(argc, argv)) {
        std::cerr << "Invalid args" << std::endl;
        return -1;
    }

    auto neuropia = NeuropiaSimple::create("");
   
    if(argparse.paramCount() <= 1) {
        std::cerr << "neuropia_verify NETWORK_FILE <DATA> <LABELS>" << std::endl;
        return 1;
    }

    const auto header = Neuropia::isValidFile(argparse.param(1));
    if(!header) {
        std::cerr << "Corrupted file: " << argparse.param(1) << std::endl;
        return 3;
    }

    const auto sizes = NeuropiaSimple::load(neuropia, argparse.param(1));
    if(!sizes) {
         std::cerr << "load failed" << std::endl;
            return 2;
    } 

    std::cout << 
    "Network loaded\nin:" << sizes->in_layer << 
    " out:" << sizes->out_layer << 
    " layers: " << header->layers  << 
    " mem:"  << NeuropiaSimple::network(neuropia).consumption(true) << 
    " type:" << Neuropia::to_string(header->saveType) <<  std::endl;
    /*for(const auto& [p, v] : NeuropiaSimple::params(neuropia)) {
        std::cout << p << "->" << v[1] << std::endl;
    }*/


    if(argparse.paramCount() > 2) {
        
        if(!NeuropiaSimple::setParam(neuropia, "ImagesVerify", argparse.param(2))) {
            std::cerr << "Bad parameter DATA" << std::endl;
            return 1;
        }

        if(!NeuropiaSimple::setParam(neuropia, "LabelsVerify", argparse.param(3))) {
            std::cerr << "Bad parameter LABELS" << std::endl;
            return 1;
        }

        const auto v = NeuropiaSimple::verify(neuropia);
        std::cout << "verify index:" << v << std::endl; 
    }

  
    return 0;
}


