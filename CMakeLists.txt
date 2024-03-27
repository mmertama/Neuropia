cmake_minimum_required (VERSION 3.15)

project(neuropiaML VERSION 1.0.1)

file(CREATE_LINK ${CMAKE_SOURCE_DIR}/wasm/mnist ${CMAKE_SOURCE_DIR}/data SYMBOLIC) # tests expects one location, WASM another, copy not ok :-D

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT NOLIB)
  add_subdirectory(neuropialib)
endif()

if(NOT NOAPP)
  add_subdirectory(app)
endif()

if(NOT NOTEST)
  add_subdirectory(test)
endif()

if(NOT NOAPP)
  add_subdirectory(verify)
endif()

if(GUTIL)
  add_subdirectory(utils/idxview)
endif()

if(EXAMPLES)
  add_subdirectory(example/romtest)
  add_subdirectory(example/compare)
endif()

