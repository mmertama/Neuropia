#include "argparse.h"
#include "neuropia_simple.h"


int main(int argc, char* argv[]) {

    ArgParse argparse;

    if(!argparse.set(argc, argv)) {
        std::cerr << "Invalid args" << std::endl;
        return -1;
    }

    auto neuropia = NeuropiaSimple::create("");
   

    if(argparse.paramCount() < 4) {
        std::cerr << "neuropia DATA LABELS OUTPUT <PARAMS>" << std::endl;
        std::cerr << "Where params is KEY=VALUE:" << std::endl;
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

    NeuropiaSimple::save(neuropia, argparse.param(3));

    return 0;
}


