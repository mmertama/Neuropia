#!/bin/bash
if [[ "$OSTYPE" == "darwin"* ]]; then
    BIN=builds/build-Neuropia-Desktop_Qt_5_12_0_clang_64bit3-Release/Neuropia
else
    BIN=../build-Neuropia-Desktop_Qt_5_12_0_GCC_64bit3-Release/Neuropia
fi
for fname in tests/tests*.txt; do
    echo ${fname}
    ${BIN} ${fname} -v -r data/mnist
    if [ "$?" -ne 0 ]; then
        echo "${fname} failed";
        exit 1;
    fi
done
