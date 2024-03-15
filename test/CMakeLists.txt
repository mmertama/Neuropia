cmake_minimum_required (VERSION 3.15)

project (neuropia_test)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(DIR ${CMAKE_SOURCE_DIR})

include_directories(${DIR}/include)

find_package (Threads)
add_executable(${PROJECT_NAME}
    main.cpp
    testports.cpp
    ${DIR}/src/idxreader.cpp
    ${DIR}/src/neuropia.cpp
    ${DIR}/src/utils.cpp
    ${DIR}/src/params.cpp
    ${DIR}/src/trainerbase.cpp
    ${DIR}/src/trainer.cpp 
    ${DIR}/src/verify.cpp
    ${DIR}/src/paralleltrain.cpp 
    ${DIR}/src/evotrain.cpp 
    ${DIR}/src/argparse.cpp
)

include (../compiler.cmake)
SET_COMPILER_FLAGS()

target_link_libraries (${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})




