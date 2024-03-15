cmake_minimum_required (VERSION 3.15)

project (neuropialib)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(DIR ${CMAKE_SOURCE_DIR})

include_directories(${DIR}/include)


add_library(${PROJECT_NAME} 
    neuropialib.h    
    ${DIR}/src/neuropia.cpp
)

include (../compiler.cmake)
SET_COMPILER_FLAGS()





