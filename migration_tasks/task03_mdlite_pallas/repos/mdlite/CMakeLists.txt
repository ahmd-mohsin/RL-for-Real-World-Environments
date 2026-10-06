cmake_minimum_required(VERSION 3.24)
project(mdlite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 89 120)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(mdlite SHARED
  src/config.cpp
  src/system.cpp
  src/capi.cpp
  src/neighbor.cu
  src/nonbonded.cu
  src/pme.cu
  src/bonded.cu
  src/constraints.cu
  src/integrate.cu
  src/simulation.cu)
# IEEE-exact double arithmetic: no fused multiply-add contraction, so results do
# not depend on how the compiler schedules the arithmetic
target_compile_options(mdlite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false> $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
target_link_libraries(mdlite PRIVATE CUDA::cudart_static CUDA::cufft)
set_target_properties(mdlite PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON
                      BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}" INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
install(TARGETS mdlite LIBRARY DESTINATION .)
