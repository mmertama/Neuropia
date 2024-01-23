#!/bin/bash

if [ ! -d "$1" ]; then
    echo "sources ${1} not found"
fi

if [ ! -d "$2" ]; then
    echo "target ${2} not found"
fi

CC=$(which cc)
CXX=$(which g++)

echo "Using ${CC} and ${CXX}"

cmake -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -S $1 -B $2 -DNOTEST=1 -DNOLIB=1
cmake --build . --config Release 

