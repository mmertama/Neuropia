#!/bin/bash

if [ ! -d "$1" ]; then
    echo "sources ${1} not found"
fi

if [ ! -d "$2" ]; then
    echo "target ${2} not found"
fi   

pushd $2
cmake -S $1 -B $2 -DNOTEST=1 -DNOLIB=1
cmake --build . --config Release 
popd
