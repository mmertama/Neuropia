emcc  -I../include -o neuropia_t.js \
      --bind -s SAFE_HEAP=0 -s WASM=1 -O3 -s INVOKE_RUN=0 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s TOTAL_MEMORY=67108864 --preload-file mnist -std=c++14 \
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
#python ${EMSDK}/fastcomp/emscripten/tools/file_packager.py mnist.data --preload mnist --js-output=mnist_data.js

#emcc  -I../include -o neuropia_t.js \
#      --bind -s SAFE_HEAP=0 -s WASM=1 -O0 -s INVOKE_RUN=0 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s TOTAL_MEMORY=67108864 --preload-file mnist -std=c++14 --source-#map-base http://localhost:8000/ -g4 \
#    ../src/idxreader.cpp \
#    ../src/neuropia.cpp \
#    ../src/utils.cpp \
#    ../src/testports.cpp \
#    ../src/params.cpp \
#    ../src/trainerbase.cpp \
#    ../src/trainer.cpp  \
#    ../src/verify.cpp \
#    ../src/paralleltrain.cpp \
#    ../src/evotrain.cpp \
#    ../src/neuropia_simple.cpp




