set(TSR_DEVICE_SOURCES
  device/error.cu
  device/memory.cu
  prims/scan.cu
  prims/reduce.cu
  prims/histogram.cu
  prims/radix_sort.cu
  prims/bitonic_sort.cu
  prims/compact.cu
  geom/camera.cu
  geom/projection.cu
  geom/mesh.cu
  bin/tile_intersect.cu
  bin/mesh_bins.cu
  raster/mesh_raster.cu
  raster/splat_raster.cu
  encode/hashgrid.cu
  encode/mlp.cu
  render/surface.cu
  render/renderer.cu
  post/tonemap.cu
  scene/scene.cu
  io/image.cu
  sched/frame_scheduler.cu
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

add_library(tessera STATIC ${TSR_DEVICE_SOURCES} ${TSR_HOST_SOURCES})
target_include_directories(tessera PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(tessera PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(tessera PUBLIC Threads::Threads)
tsr_target_warnings(tessera)
if(TSR_ENABLE_CUDA)
  set_target_properties(tessera PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(tessera PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
endif()
