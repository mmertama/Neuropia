#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include "neuropia.h"
#include "utils.h"

void testGates(const std::string& name, const std::vector<std::tuple<Neuropia::ValueVector, Neuropia::ValueVector>>& data) {

    auto network = Neuropia::Layer(2);
    network.join(2);
    network.join(1);

    network.randomize();
    constexpr auto runs = 50000;


    const auto seed =
#ifndef RANDOM_SEED
            static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
#else
    RANDOM_SEED
#endif
    ;
    std::default_random_engine gen(seed);

    for(size_t i = 0U; i  < runs ; i++) {
        const auto index = gen() % data.size();
        network.train(std::get<0>(data[index]).begin(), std::get<1>(data[index]).begin(),  0.05, 0.0);
    }

    for(const auto& d : data) {
        const auto& feed = std::get<0>(d);
        std::cout << name << " " << feed << "->" << network.feed(feed) << std::endl;
    }
}

void testLogicalPorts() {
    Neuropia::timed([]() {
        testGates("xor", {
            {{0, 0}, {0}},
            {{1, 0}, {1}},
            {{0, 1}, {1}},
            {{1, 1}, {0}}
        });
    });

    Neuropia::timed([]() {
        testGates("and", {
            {{0, 0}, {0}},
            {{1, 0}, {0}},
            {{0, 1}, {0}},
            {{1, 1}, {1}}
        });
    });

    Neuropia::timed([]() {
        testGates("or", {
            {{0, 0}, {0}},
            {{1, 0}, {1}},
            {{0, 1}, {1}},
            {{1, 1}, {1}}
        });
    });

    Neuropia::timed([]() {
        testGates("nand", {
            {{0, 0}, {1}},
            {{1, 0}, {0}},
            {{0, 1}, {0}},
            {{1, 1}, {0}}
        });
    });

    Neuropia::timed([]() {
        testGates("nor", {
            {{0, 0}, {1}},
            {{1, 0}, {1}},
            {{0, 1}, {1}},
            {{1, 1}, {0}}
        });
    });
}
