set(NEUROPIA_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
include(${NEUROPIA_DIR}/cmake/neuropia.cmake)

set(NEUROPIA_SOURCES
    ${NEUROPIA_DIR}/neuropialib/neuropialib.h     
    ${NEUROPIA_DIR}/src/neuropia.cpp
)

set(NEUROPIA_INCLUDE
    ${NEUROPIA_DIR}/neuropialib/neuropialib.h     
)

set(NEUROPIA_INCLUDE_DIR
    ${NEUROPIA_DIR}/neuropialib 
    ${NEUROPIA_DIR}/include 
)




function (make_neuropia TARGET_DIR)
    
    find_program(SYS_CMAKE cmake REQUIRED)
    
    message("${TARGET_DIR} from ${NEUROPIA_DIR}, using ${SYS_CMAKE}")
    
    file(MAKE_DIRECTORY ${TARGET_DIR})

    if (NOT EXISTS ${NEUROPIA_DIR}/neuropialib/make_neuropia.sh)
        message(FATAL_ERROR "${NEUROPIA_DIR}/neuropialib/make_neuropia.sh not found!")
    endif()

    execute_process(
        COMMAND ${NEUROPIA_DIR}/neuropialib/make_neuropia.sh ${NEUROPIA_DIR} ${TARGET_DIR}
        WORKING_DIRECTORY ${TARGET_DIR}
        COMMAND_ECHO STDOUT
        ECHO_OUTPUT_VARIABLE
        ECHO_OUTPUT_VARIABLE
        COMMAND_ERROR_IS_FATAL ANY)

endfunction()
