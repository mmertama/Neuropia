#include "argparse.h"
#include "neuropia_simple.h"
#include "utils.h"
#include "idxreader.h"
#include <iostream>


int main(int argc, char* argv[]) {

    ArgParse argparse;
    argparse.addOpt('i', "index", true);
    argparse.addOpt('c', "count", true);
    if(!argparse.set(argc, argv)) {
        std::cerr << "Invalid args" << std::endl;
        return -1;
    }

    auto neuropia = NeuropiaSimple::create("");
   
    if(argparse.paramCount() <= 1) {
        std::cerr << "neuropia_verify <-i INDEX> <-c COUNT> NETWORK_FILE <DATA> <LABELS>" << std::endl;
        return 1;
    }

    const auto header = Neuropia::isValidFile(argparse.param(1));
    if(!header) {
        std::cerr << "Corrupted or missing file: " << argparse.param(1) << std::endl;
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
    " layers:" << header->layers  << 
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
        
        Neuropia::IdxReader<unsigned char> idx(argparse.param(3));

        if(!idx.ok() || idx.size() == 0) {
            std::cerr << "Bad parameter LABELS" << std::endl;
            return 1;
        }

        

        if(argparse.hasOption("index")) {
            const auto index_param = argparse.option("index");
            if(!Neuropia::isnumber(index_param)) {
                std::cerr << "Bad index " << index_param << std::endl;
                return 1;
            }

            const auto index = static_cast<unsigned>(std::stoi(index_param));
            Neuropia::IdxReader<unsigned char> idx_images(argparse.param(2));
            if(idx_images.size() <= index) {
                 std::cerr << "Bad image INDEX " << index << std::endl;
                return 1;
            }
            if(idx.size() <= index) {
                 std::cerr << "Bad label INDEX " << index << std::endl;
                return 1;
            }
            const auto dims = idx_images.size(1) * idx_images.size(2);
            const auto image_data = idx_images.readAt(index, dims); 

            std::vector<double> image;
            std::transform(image_data.begin(), image_data.end(), std::back_inserter(image), [](const auto c){return c;});
            
            const auto result = NeuropiaSimple::feed(neuropia, image);
            const auto order = Neuropia::ordered(result.begin(), result.end());

            for(const auto& p : order) {
                std::cout << static_cast<int>(p.first) << " " << p.second << std::endl;
            }

            const auto expected = static_cast<unsigned>(idx.readAt(index));
            if(expected == order.front().first)
                std::cout << "The " << expected << " looks ok!" << std::endl;
            else
                std::cout << "It get " << order.front().first  << " When expecting " << expected << std::endl;

        } else  {
            auto count = std::numeric_limits<size_t>::max();
            if(argparse.hasOption("count")) {
                 const auto count_param = argparse.option("count");
                 if(!Neuropia::isnumber(count_param)) {
                    std::cerr << "Bad count " << count_param << std::endl;
                    return 1;
                }
                count = static_cast<size_t>(std::stoi(count_param));
            }
            const auto index_param = argparse.option("index");   
            const auto r = NeuropiaSimple::verify(neuropia, count);
            std::cout << "Verify match:" << std::get<1>(r) << "% size:" << idx.size() << " looked:" << std::min(idx.size(), count) << " found:" << std::get<0>(r)  << std::endl; 
        }
    }

  
    return 0;
}


