# Neuropia
Minimalistic C++ Neural Network library.

The comprehensive introduction [article series](https://www.insta.fi/en/expert-blog/road-to-neuropia) that closely elaborates Neuropia.

Markus Mertama 2019, 2024

## Ways to use

#### Neuropia.h 
Includes a Full API where you can tune training parameters and create new activation functions etc.

#### Neuropia Simple
See neuropia_simple.h to get started, it is a high level wrapper to construct a network. You still can adjust most
of the things by setting parameters.

#### Use as command line application
`neuropia` trains a network for classification. 1st parameter is image (or such) data in 3-dimensional [IDX](https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html),
second 1-dimensional labels, and 3rd is output name. The training parameters (run `neuropia` without parameters to see available parameters) are added after 3rd parameter as KEY=VALUE pairs. Build and `neuropia` app create networks for you.

e.g.
```
$neuropia mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte neuropia.bin Iterations=10000 Hard=true

```

`neuropia_test` is used to run network efficency evaluations (see Testing below).

#### Embedded libraries
Minimal API in `neuropia_lib` folder to utilize pre-trained network with minimal resources, for example in embeded systems (MCUs). 

The API:

```cpp 
std::optional<Sizes> Neuropia::Network::load(const Bytes& bytes);
std::optional<Sizes> Neuropia::Network::load(const uint8_t& bytes, size_t sz);
Values Neuropia::Network::feed(const Values& input) const;
```

* Load function loads the trained network as a input (i.e. the binary file `neuropia` app created).
* Feed function takes the input layer as an input and returns output layer.

When network code is backed in sources it can be used as

```cpp
#include "neuropia_bin.h" // defines a neuropia_bin, generated by bin2code.py
...
Neuropia::Network network;
network.load(neuropia_bin, sizeof(neuropia_bin));

```

##### Hints for CMake

```
FetchContent_Declare(
   neuropia
    GIT_REPOSITORY https://github.com/mmertama/Neuropia.git)

FetchContent_MakeAvailable(neuropia)

include(${neuropia_SOURCE_DIR}/cmake/neuropialib.cmake)

...

target_sources(${PROJECT_NAME} PRIVATE
    ...
    ${NEUROPIA_SOURCES}
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ...
    ${NEUROPIA_INCLUDE_DIR}
    )

    
```

You can add network generation as part of a build process

```
make_neuropia("${CMAKE_CURRENT_BINARY_DIR}/neuropia/build")
find_program(NEUROPIA neuropia PATHS "${CMAKE_CURRENT_BINARY_DIR}/neuropia/build/app" REQUIRED)


target_sources(${PROJECT_NAME} PRIVATE
...
    ${NEUROPIA_SOURCES}
    ${CMAKE_BINARY_DIR}/neuropia_bin.h
    )

# generate network
add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/neuropia.bin"
    COMMAND ${NEUROPIA} "${MNIST_DATA_IMAGES}" "${MNIST_DATA_LABELS}" "${CMAKE_BINARY_DIR}/neuropia.bin"
    DEPENDS "${MNIST_DATA_IMAGES}" "${MNIST_DATA_LABELS}"
)

# make it as header you can include
add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/neuropia_bin.h"
    COMMAND ${Python3_EXECUTABLE} ${neuropia_SOURCE_DIR}/utils/bin2code.py "${CMAKE_BINARY_DIR}/neuropia.bin"  "${CMAKE_BINARY_DIR}/neuropia_bin.h" "neuropia_bin"
    DEPENDS "${CMAKE_BINARY_DIR}/neuropia.bin"
)

# some cmake dependency mambo
set_source_files_properties(${CMAKE_BINARY_DIR}/neuropia_bin.h PROPERTIES GENERATED TRUE)
add_custom_target(neuropia_generation ALL DEPENDS "${CMAKE_BINARY_DIR}/neuropia_bin.h")
set_source_files_properties(src/hwr.cpp PROPERTIES  OBJECT_DEPENDS neuropia_generation)
add_dependencies(${PROJECT_NAME} neuropia_generation)

```

## Build
Use cmake

Supports Windows MSCV, GCC and Clang, Emscripten (Web Assembly)

## Testing
For testing copy the Mnist data from [huggingface](https://huggingface.co/datasets/mnist)

Extract GZ files data/mnist folder (tests_all.sh assume that folder, but otherwise pick freely)

For `neuropia_test`, there is a commandline option -r to tell the folder. 

## Run test
Easiest way to run is to use tests. Note that for tests data files has to be renamed.
E.g. you have use CMakeLists.txt in the separate build folder, then if mnist data is located in the source folder, you can do something like:
`./neuropia_test ../neuropia/tests/tests2.txt -v -r ../neuropia/data/mnist`



