# Neuropia
Markus Mertama 2019
## Build
Use neuropia.pro or CMakeLists.txt

## Data
Copy Mnist data from http://yann.lecun.com/exdb/mnist/
Extract GZ files data/mnist folder (tests_all.sh assume that folder, but otherwise pick freely)

## Run
Easiest way to run is to use tests
E.g. you have use CMakeLists.txt in the separate build folder, then if mnist data
is located in source folder, you can do something like:
`./neuropia ../neuropia/tests/tests2.txt -v -r ../neuropia/data/mnist`


