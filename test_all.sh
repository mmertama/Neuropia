#!/bin/bash
BIN=$(find build/release -name neuropia_test)
DATA=$(find . -name train-images-idx3-ubyte -exec dirname {} \;)
for fname in tests/tests*.txt; do
    echo ${fname}
    ${BIN} ${fname} -v -r ${DATA}
    if [ "$?" -ne 0 ]; then
        echo "${fname} failed";
        exit 1;
    fi
done
