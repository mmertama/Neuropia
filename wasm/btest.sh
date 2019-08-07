emcc  -I../include -o neuropia_t.js \
      --bind -s VERBOSE=1 -s SAFE_HEAP=0 -s WASM=1 -O0 -s INVOKE_RUN=0 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -std=c++14 \
    ../src/idxreader.cpp \
    ../src/neuropia.cpp \
    ../src/utils.cpp \
    ../src/testports.cpp \
    ../src/params.cpp \
    ../src/trainerbase.cpp \
    ../src/trainer.cpp  \
    ../src/verify.cpp \
    ../src/paralleltrain.cpp \
    ../src/evotrain.cpp \
    ../src/neuropia_simple.cpp


