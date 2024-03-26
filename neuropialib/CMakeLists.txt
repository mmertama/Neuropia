cmake_minimum_required (VERSION 3.21)

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

if(LINUX)
    set(DOXYGEN TRUE)
    set(SPELL TRUE)
endif() 

if(DOXYGEN)
  include(FetchContent)
  FetchContent_Declare(
      cmakedoc
      GIT_REPOSITORY https://github.com/mmertama/cmakedoc.git
      GIT_TAG        main
  )
  FetchContent_MakeAvailable(cmakedoc)
  include(${cmakedoc_SOURCE_DIR}/cmakedoc.cmake)
  set(CMAKEDOC_DOXYGEN_DOCUMENTS
      ${CMAKE_SOURCE_DIR}/include/idxreader.h
      ${CMAKE_SOURCE_DIR}/include/neuropia.h
      ${CMAKE_SOURCE_DIR}/include/neuropia_simple.h
      ${CMAKE_SOURCE_DIR}/include/neuropia_feed.h
      neuropialib.h
  )
  set(DOXYGEN_PROJECT_NAME "Neuropia")
  add_doxygen(${PROJECT_NAME})


  if(SPELL)
      set(CMAKEDOC_SPELL_DICTIONARY ${CMAKE_SOURCE_DIR}/aspell/neuropia_words.txt)
      set(CMAKEDOC_SPELL_EXCLUDE_DIRS
          build
          aspell
          modules
          .git
          tests
          src
          cmake-build-debug)
      add_spellcheck(${PROJECT_NAME})    
  endif()    
endif()



