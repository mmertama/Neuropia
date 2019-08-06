
cmake_minimum_required(VERSION 3.5.1)

project (neuropia)

option(JS_ONLY "Compiles to native JS (No WASM)" OFF)


set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++14 -O3)

include_directories(../include)

set(CMAKE_C_COMPILER "emcc")

file(GLOB SOURCES ../src/*.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/public")

add_library(neuropia SHARED ${SOURCES})

set(EMCC_LINKER_FLAGS "-s SAFE_HEAP=1 --bind -s WASM=1 -O3 -o ../index.js -s LEGACY_GL_EMULATION=0  -s GL_UNSAFE_OPTS=0 --pre-js pre-module.js --post-js post-module.js -s ASSERTIONS=1 -s INVOKE_RUN=0  -std=c++14 -s FULL_ES3=1 --preload-file shaders --preload-file extern --use-preload-plugins")

set(CMAKE_REQUIRED_FLAGS "${EMCC_LINKER_FLAGS}")

set_target_properties(neuropia PROPERTIES LINK_FLAGS "${EMCC_LINKER_FLAGS}")

