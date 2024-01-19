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

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)



