cmake_minimum_required (VERSION 3.1)

project (neuropia)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT MSVC)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 14)

include_directories(include)

find_package (Threads)

add_executable(neuropia
    src/idxreader.cpp
    src/main.cpp
    src/neuropia.cpp
    src/utils.cpp
    src/testports.cpp
    src/params.cpp
    src/trainerbase.cpp
    src/trainer.cpp 
    src/verify.cpp
    src/paralleltrain.cpp 
    src/evotrain.cpp 
    src/argparse.cpp
)

target_link_libraries (neuropia ${CMAKE_THREAD_LIBS_INIT})


