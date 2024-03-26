
cmake_minimum_required(VERSION 3.21.0)

project (neuropia) #  neuropia as referred by this name 

include (../compiler.cmake)
SET_COMPILER_FLAGS()

include_directories(../include)

add_executable(${PROJECT_NAME}
    ../src/idxreader.cpp
    ../src/neuropia.cpp
    ../src/utils.cpp
    ../src/params.cpp
    ../src/trainerbase.cpp
    ../src/trainer.cpp 
    ../src/verify.cpp
    ../src/neuropia_simple.cpp
    neuropia_wasm.cpp  
)

set(EMCC_LINKER_FLAGS "-o neuropia.js \
      --bind --no-heap-copy -s ASSERTIONS=1 -s SAFE_HEAP=0 -s SAFE_HEAP_LOG=0 -s ALIASING_FUNCTION_POINTERS=0 -s WARN_UNALIGNED=0 -s WASM=1 -O3 -s INVOKE_RUN=0 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=0 -s TOTAL_MEMORY=1024MB --preload-file mnist -s EXIT_RUNTIME=1")

set(CMAKE_REQUIRED_FLAGS "${EMCC_LINKER_FLAGS}")

set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${EMCC_LINKER_FLAGS}")


