#ifndef PSRAND_H
#define PSRAND_H


#include <limits>
#include <algorithm>

namespace NTest {

/*
 *
 * Javascript counterparts

 let x0 = new Number(Math.floor(2147483647 / 2));

function ipsrand(){
    x0 = Math.floor(((x0 * 16807) >>> 0) % 2147483647);
    return x0;
};

function psrand(min, max) {
    ipsrand();
    return min + (max - min) * (x0 - min) / 2147483647;
}
 *
 *
 */


inline unsigned ipsrand(int any = 0){
    static constexpr unsigned seed = 2147483647 / 2;
    static unsigned  x0 = seed;
    if(any < 0) {
        x0 = seed;
        return seed;
    }
    x0 = (x0 * 16807) % 2147483647;
    return x0;
};

template <typename T>
T psrand(T min, T max) {
    const auto x0 = ipsrand();
    return min + (max - min) * (static_cast<T>(x0) - min) / 2147483647.0;
}


}

#endif // PSRAND_H
