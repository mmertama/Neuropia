# Neuropia
C++ Neural Network library.

The comprehensive introduction [article series](https://www.insta.fi/en/expert-blog/road-to-neuropia) that closely elaborates Neuropia.

To code, see neuropia_simple.h to get started, it is a high level wrapper to construct a network.

Markus Mertama 2019
## Build
Use neuropia.pro or CMakeLists.txt

Supports Windows MSCV, GCC and Clang, Emscripten

## Data
For testing copy the Mnist data from [huggingface](https://huggingface.co/datasets/mnist)

Extract GZ files data/mnist folder (tests_all.sh assume that folder, but otherwise pick freely)

There is a commandline option -r to tell the folder. . Note that for tests data files has to be renamed.

## Run
Easiest way to run is to use tests. Note that for tests data files has to be renamed.
E.g. you have use CMakeLists.txt in the separate build folder, then if mnist data is located in the source folder, you can do something like:
`./neuropia ../neuropia/tests/tests2.txt -v -r ../neuropia/data/mnist`


