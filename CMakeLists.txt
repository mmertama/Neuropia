cmake_minimum_required (VERSION 3.15)

project(neuropiaML)

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
