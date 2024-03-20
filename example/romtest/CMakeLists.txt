cmake_minimum_required (VERSION 3.15)

project (rom_test)

find_package (Python3 REQUIRED)

set(NEUROPIA_DIR ${CMAKE_SOURCE_DIR})
include("${NEUROPIA_DIR}/cmake/neuropia.cmake")

# This may not work an 1st build, build 1st without EXAMPLES, then rebuild?
# ...or setup dependencies and do find buildtime
set(NEUROPIA "${CMAKE_BINARY_DIR}/app/neuropia")
  
set(BIN_FOLDER "${CMAKE_BINARY_DIR}/example")

# Train
add_custom_command(
    OUTPUT "${BIN_FOLDER}/neuropia.bin"
    COMMAND ${NEUROPIA} -d float "${MNIST_DATA_IMAGES}" "${MNIST_DATA_LABELS}" "${BIN_FOLDER}/neuropia.bin" Iterations=3000 Classes=10
    COMMENT "Process ${MNIST_DATA_IMAGES} ${MNIST_DATA_LABELS}"
    DEPENDS "${MNIST_DATA_IMAGES}" "${MNIST_DATA_LABELS}"
    )
    

# Bake results in
add_custom_command(
    OUTPUT "${BIN_FOLDER}/neuropia_bin.h"
    COMMAND ${Python3_EXECUTABLE} ${NEUROPIA_DIR}/utils/bin2code.py "${BIN_FOLDER}/neuropia.bin" "${BIN_FOLDER}/neuropia_bin.h" "neuropia_bin"
    DEPENDS "${BIN_FOLDER}/neuropia.bin"
)

set_source_files_properties(${BIN_FOLDER}/neuropia_bin.h
                            PROPERTIES GENERATED TRUE)

if(NOT TARGET neuropia_generation)
    add_custom_target(neuropia_generation ALL DEPENDS "${BIN_FOLDER}/neuropia_bin.h")
endif()

set_source_files_properties(src/main.cpp
                            PROPERTIES  OBJECT_DEPENDS neuropia_generation)


add_executable(${PROJECT_NAME}
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/idxreader.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp       # percentage
    ${CMAKE_SOURCE_DIR}/src/neuropia.cpp    # utils
    ${CMAKE_SOURCE_DIR}/src/argparse.cpp
    "${BIN_FOLDER}/neuropia_bin.h"
    )

target_include_directories(${PROJECT_NAME} PRIVATE 
  ${BIN_FOLDER}
  ${CMAKE_SOURCE_DIR}/include
  )

target_compile_definitions(${PROJECT_NAME} PRIVATE NEUROPIA_TYPE=float)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_dependencies(${PROJECT_NAME} neuropia_generation)
add_dependencies(neuropia_generation neuropia)





