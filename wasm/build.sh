rm libneuropia.a
rm CMakeCache.txt
rm -rf CMakeFiles
rm MakeFile 
emcmake cmake CMakeLists.txt
emmake make
emcc -v -s VERBOSE=1 --bind libneuropia.a -o neuropia.js


