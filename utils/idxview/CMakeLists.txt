cmake_minimum_required (VERSION 3.20)

project (idxview)

find_package(gempyre REQUIRED)

include(gempyre)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}
    main.cpp
    ../../src/idxreader.cpp
    gui.html
    )

gempyre_add_resources(PROJECT ${PROJECT_NAME}
    TARGET ${CMAKE_BINARY_DIR}/inc/resources.h 
    SOURCES gui.html)    

target_include_directories(${PROJECT_NAME}
    PRIVATE ../../include
    PRIVATE ${CMAKE_BINARY_DIR}/inc
    )

target_link_libraries(${PROJECT_NAME} gempyre::gempyre)

