
cmake_minimum_required(VERSION 3.5.1)

set(NAME neuropia)

project (${NAME})

#option(JS_ONLY "Compiles to native JS (No WASM)" OFF)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++14 -O3)

include_directories(../include)

#set(CMAKE_C_COMPILER "emcc")
#set(CMAKE_CXX_COMPILER "em++")

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/public")

add_executable(${NAME}
    ../src/idxreader.cpp
    ../src/neuropia.cpp
    ../src/utils.cpp
    ../src/testports.cpp
    ../src/params.cpp
    ../src/trainerbase.cpp
    ../src/trainer.cpp 
    ../src/verify.cpp
    ../src/paralleltrain.cpp 
    ../src/evotrain.cpp
    ../src/neuropia_simple.cpp  
)


set(EMCC_LINKER_FLAGS "-v --bind -s VERBOSE=1 -s SAFE_HEAP=0 -s WASM=1 -O0 --pre-js pre-module.js --post-js post-module.js -s ASSERTIONS=1 -s INVOKE_RUN=0 -s ERROR_ON_UNDEFINED_SYMBOLS=1 -std=c++14")

set(CMAKE_REQUIRED_FLAGS "${EMCC_LINKER_FLAGS}")

set_target_properties(${NAME} PROPERTIES LINK_FLAGS "${EMCC_LINKER_FLAGS}")


