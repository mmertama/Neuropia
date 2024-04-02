# Neuropia
Minimalistic C++ Neural Network library.

The comprehensive introduction [article series](https://www.insta.fi/en/software-consulting/news/expert-blogs/road-to-neuropia/) that closely elaborates Neuropia.

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

For embedded system the compiled network size can be reduced with -d float parameter. Even the network is calculated using REALTYPE=double (default)   

e.g.
```
$neuropia mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte neuropia.bin Iterations=10000 Hard=true Classes=10

```

`neuropia_test` is used to run network efficiency evaluations (see Testing below).

#### Embedded libraries
Minimal API in `neuropia_lib` folder to utilize pre-trained network with minimal resources, for example in embedded systems (MCUs). 

The API:

```cpp 
std::optional<Sizes> Neuropia::Network::load(const Bytes& bytes);
std::optional<Sizes> Neuropia::Network::load(const uint8_t& bytes, size_t sz);
const Values& feed(const ITType& begin, const ITType& end) const;
```

* Load function loads the trained network data as a input (i.e. the binary file `neuropia` app created) and constructs the network. 
* Feed function takes the input layer as an input and returns output layer. It is granted not to allocate any memory.

When network code is backed in sources it can be used as

```cpp
#include "neuropia_bin.h" // defines a neuropia_bin, generated by bin2code.py
...
Neuropia::Network network;
network.load(neuropia_bin, sizeof(neuropia_bin));

```

#### Feeder

When memory is tight (e.g. embedded devices, or tons of parallel networks needed) 
the `neuropia_feed.h` implements a header only Neuropia::Feed class that loads a pre-trained network from the included file (see above) without any dynamic allocation. It is a templated class reads values directly from the included header, since all the parameters are known at build time, the network itself is constructed at build time. Cost of radically decreased memory consumption is a slight performance hit.


Using is simple:

Add the pre-compiled data as a header:

```cmake

# Bake results in
add_custom_command(
    OUTPUT "${BIN_FOLDER}/neuropia_bin.h"
    COMMAND ${Python3_EXECUTABLE} ${NEUROPIA_DIR}/utils/bin2code.py "${BIN_FOLDER}/neuropia.bin" "${BIN_FOLDER}/neuropia_bin.h" "neuropia_bin"
    DEPENDS "${BIN_FOLDER}/neuropia.bin"
)


```

Call Neuropia::Feed::feed with your data:


```cpp

#include "neuropia_bin.h"

using NeuropiaFeed = Neuropia::Feed<neuropia_bin, sizeof(neuropia_bin)>;

...

  const auto outputs = NeuropiaFeed::feed(inputs.begin(), inputs.end());
....          

```

#### Playground
The [Neuropia Live](https://mmertama.github.io/Neuropia/neuropia.html) is running on WebAssembly.

## Build
Use cmake

Supports Windows MSVC, GCC and Clang, Emscripten (Web Assembly)

#### Hints for CMake

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


#### Using Neuropia Simple

Basically just create + train/load/save based on parameters given. Minimum parameters are input images and labels filenames.
The default parameters are defined in´ [default.h](https://github.com/mmertama/Neuropia/blob/master/include/default.h).


```cpp

// create a Neuropia
auto neuropia = NeuropiaSimple::create("");
// most of parameters you can use defaults
NeuropiaSimple::setParam(neuropia, "Images", "images.idx");
NeuropiaSimple::setParam(neuropia, "Labels", "labels.idx");
// Set output layers default is 10, input layers it get from images dimensions
NeuropiaSimple::setParam(neuropia, "Classes", 10); 
NeuropiaSimple::setParam(neuropia, "Iterations", 10000);
// do train
NeuropiaSimple::train(neuropia, NeuropiaSimple::TrainType::Basic);
NeuropiaSimple::save(neuropia, "neuropia.bin" , Neuropia::SaveType::Double);


```

### Utils (in the utils folder)
 
* idxview 
    *  to view idx file content 
    * `idxview IMAGES.idx LABELS.idx INDEX`
    * Uses [Gempyre](https://github.com/mmertama/Gempyre) 
* bin2code.py
    * Generates C++ header from a binary
    * `bin2code.py BIN_FILE HEADER_FILE C_ARRAY_NAME`
* mit2idx.py
    * `mit2idx.py by_class.zip TARGET_FOLDER NAME_PREFIX IMAGE_WIDTH IMAGE_HEIGHT`
    * Convert 'mit' to idx (in practice https://data.world/nist/nist-handprinted-form-charactezip to idx)


##### Out of memory when building
Compiling the `neuropia_feed.h` uses a lot of memory. My Ubuntu 32G with 2G swap was far too small. Even
for the most simple (rom_text) network I has to increase the swap size to 25G! You can temporary add
another swapfile with following snippet:

```bash
 
 $ sudo fallocate -l 25G /swapfile2
 $ sudo chmod 600 /swapfile2
 $ sudo mkswap /swapfile2
 $ sudo swapon /swapfile2

```

On next reboot thaw swapfile should be removed.


## Documents
[Doxygen documentation](https://mmertama.github.io/Neuropia/docs/)

## Testing
For testing copy the Mnist data from [huggingface](https://huggingface.co/datasets/mnist)

Extract GZ files data/mnist folder (tests_all.sh assume that folder, but otherwise pick freely)

For `neuropia_test`, there is a command line option -r to tell the folder. 

## Run test
Easiest way to run is to use tests. Note that for tests data files has to be renamed.
E.g. you have use CMakeLists.txt in the separate build folder, then if mnist data is located in the source folder, you can do something like:
`Neuropia/build$ ./test/neuropia_test ../tests/tests2.txt -v -r ../data/mnist`



