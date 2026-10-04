tsr_device_sources(prims_bench.cu)
add_executable(prims_bench prims_bench.cu)
target_link_libraries(prims_bench PRIVATE tsr_prims)
tsr_target_warnings(prims_bench)
