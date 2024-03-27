#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -z "$1" ]; then
    PARAMS="-DDEBUG_FLAGS=\"$1\""
fi    

mkdir -p $SCRIPT_DIR/../build
mkdir -p $SCRIPT_DIR/../build/wasm
pushd $SCRIPT_DIR/../build/wasm
emcmake cmake $SCRIPT_DIR/CMakeLists.txt $PARAMS
emmake make
popd
