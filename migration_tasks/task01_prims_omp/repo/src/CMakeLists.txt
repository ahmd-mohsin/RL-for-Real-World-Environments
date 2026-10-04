set(TSR_DEVICE_SOURCES
  device/error.cu
  device/memory.cu
  prims/scan.cu
  prims/reduce.cu
  prims/histogram.cu
  prims/radix_sort.cu
  prims/bitonic_sort.cu
  prims/compact.cu
)

set(TSR_HOST_SOURCES)

if(NOT TSR_ENABLE_CUDA)
  list(APPEND TSR_HOST_SOURCES
    device/emu/runtime.cpp
    device/emu/fiber_context.cpp
    device/emu/cuda_runtime.cpp
  )
endif()

tsr_device_sources(${TSR_DEVICE_SOURCES})

add_library(tsr_prims STATIC ${TSR_DEVICE_SOURCES} ${TSR_HOST_SOURCES})
target_include_directories(tsr_prims PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(tsr_prims PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(tsr_prims PUBLIC Threads::Threads)
tsr_target_warnings(tsr_prims)
if(TSR_ENABLE_CUDA)
  set_target_properties(tsr_prims PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(tsr_prims PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
endif()
