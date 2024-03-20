#include "neuropia_feed.h"
#include "neuropia_bin.h"
#include "utils.h"
#include <iostream>

using NeuropiaFeed = Neuropia::Feed<neuropia_bin, sizeof(neuropia_bin)>;

int main(int argc, char* argv[]) {
    
    std::cout << "bytes: " << sizeof(neuropia_bin) << 
    "\nlayers: " << static_cast<int>(NeuropiaFeed::layer_count()) << 
    "\nparameters: " << static_cast<int>(NeuropiaFeed::parameter_count()) << std::endl;
    for(auto i = 0U; i < NeuropiaFeed::parameter_count(); ++i) {
        const auto& [key, value] = NeuropiaFeed::parameter(i) ;
        std::cout << i + 1 << ") " << key << "->" << value << std::endl;
    }

    for(auto i = 0U; i < NeuropiaFeed::layer_count(); ++i) {
        const auto& [sz, af] = NeuropiaFeed::layer_info(i);
        std::cout << "L:" << i << " " << sz << " " << af  << " " << std::endl;
    }

    if(argc < 2) {
        std::cerr << "Expect NETWORK" << std::endl;
        return 1;
    }

    const auto res = Neuropia::load(argv[1]);

    if(!res) {
        std::cerr << "Cannot load NETWORK" << std::endl;
        return 1;
    }

    const auto& [network, params] = *res;

    std::cout << "Loaded..." << std::endl;

    int i = 0;
    for(const auto& [key, value] : params) {
        std::cout << ++i << ") " << key << "->" << value << std::endl;
    }

    i = 0;
    size_t layers_sz = 0;
    auto next = &network;
    while(next) {
         std::cout << "L:" << i << " " << next->size() << " " << next->activationFunction().name() << std::endl;
         next = next->next();
         ++layers_sz;
    }

    if(NeuropiaFeed::parameter_count() != params.size()) {
        std::cerr << "Parameter size mismatch" << std::endl;
        return 1;
    }


    for(auto i = 0U; i < NeuropiaFeed::parameter_count(); ++i) {
        const auto& [key, value] = NeuropiaFeed::parameter(i);
        const auto it = params.find(std::string(key));
        if(it == params.end()) {
            std::cerr << "Parameter name mismatch " << key << std::endl;
            return 1;
        }
        if(value != it->second) {
            std::cerr << "Parameter value mismatch " << key << " " << value << "!=" << it->second << std::endl;
            return 1;
        }
    }

    if(NeuropiaFeed::layer_count() != layers_sz) {
        std::cerr << "Layer count mismatch " << NeuropiaFeed::layer_count() << "!=" << layers_sz << std::endl;
        return 1;
    }

    next = &network;
    for(auto l = 0U; l < NeuropiaFeed::layer_count(); ++l) {
        const auto& [neuron_count, af] = NeuropiaFeed::layer_info(l);
        
        if(neuron_count != next->size()) {
            std::cerr << "Layer size mismatch " << l << " " << neuron_count << "!=" << next->size() << std::endl;
            return 1;
        }
        
        if(af != next->activationFunction().name()) {
             std::cerr << "Layer activation function mismatch " << l << " " << next->activationFunction().name() << "!=" << af << std::endl;
            return 1;
        }
        
        for(auto n = 0U; n < neuron_count; ++n) {
            
            const auto ni = NeuropiaFeed::neuron_info(l, n);
            neuropia_assert(ni);
            const auto& [weights, bias] = *ni;

            const auto neuron = (*next)[n];
            
            if(bias != neuron.bias()) {
                std::cerr << "Neuron bias mismatch " << l << "-" << n << " " << bias << "!=" << neuron.bias() << std::endl;
                return 1;
            }
            
            if(weights.size()  != neuron.size()) {
                std::cerr << "Neuron size mismatch " << l << "-" << n << " " << weights.size() << "!=" << neuron.size() << std::endl;
                return 1;
            }

            for(auto w = 0U; w < weights.size(); ++w) {
                if(weights[w] != neuron.weight(w)) {
                    std::cerr << "Neuron weight mismatch " << l << "-" << n << "-" << w << " " << weights[w] << "!=" << neuron.weight(w) << std::endl;
                    return 1;
                }
            }
        }
        next = next->next();
    }

    return 0;
}


