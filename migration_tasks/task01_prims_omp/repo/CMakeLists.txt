cmake_minimum_required(VERSION 3.24)
project(tsr_prims VERSION 0.7.0 LANGUAGES CXX)

option(TSR_ENABLE_CUDA "Compile device code with nvcc (requires the CUDA toolkit)" OFF)
option(TSR_BUILD_TESTS "Build the test suite" ON)
option(TSR_BUILD_APPS "Build command line tools" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

if(TSR_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
  endif()
endif()

include(cmake/TesseraDevice.cmake)

add_subdirectory(src)

if(TSR_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(TSR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
