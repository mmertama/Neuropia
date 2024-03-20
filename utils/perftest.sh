#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$SCRIPT_DIR/..
IMAGES=$ROOT/wasm/mnist/t10k-images-idx3-ubyte
LABELS=$ROOT/wasm/mnist/t10k-labels-idx1-ubyte
BINROOT=$ROOT/build/release
NET=$BINROOT/example/neuropia.bin
echo Rom
$BINROOT/example/romtest/rom_test $IMAGES $LABELS -q
echo Verify
$BINROOT/verify/neuropia_verify $NET $IMAGES $LABELS
echo Valgrind
mkdir -p $ROOT/build/valgrind
pushd $ROOT/build/valgrind
rm massif.out.*
PID1=$(valgrind --tool=massif $BINROOT/example/romtest/rom_test $IMAGES $LABELS -q 2>&1 | grep -o '==[0-9]\+==' | head -n1 | sed 's/==//g')
PID2=$(valgrind --tool=massif $BINROOT/verify/neuropia_verify $NET $IMAGES $LABELS 2>&1 | grep -o '==[0-9]\+==' | head -n1 | sed 's/==//g')
popd
echo P1=$PID1 P2=$PID2 done
