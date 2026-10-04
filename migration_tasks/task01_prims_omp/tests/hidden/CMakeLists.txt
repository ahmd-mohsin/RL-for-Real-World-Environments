# Hidden test suite for task01 (replaces tests/ at grading time).
# Requires the library target `tsr_prims` with its offload flags as PUBLIC
# usage requirements.
add_executable(tsr_tests test_main.cpp test_prims_hidden.cpp)
target_link_libraries(tsr_tests PRIVATE tsr_prims)
target_include_directories(tsr_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(tsr_tests PRIVATE -w)
add_test(NAME tsr_tests COMMAND tsr_tests --no-intro --duration=false)

add_executable(tsr_offload_probe offload_probe.cpp)
target_link_libraries(tsr_offload_probe PRIVATE tsr_prims ${CMAKE_DL_LIBS})
target_compile_options(tsr_offload_probe PRIVATE -w)

add_executable(tsr_perf_bench perf_bench.cpp)
target_link_libraries(tsr_perf_bench PRIVATE tsr_prims)
target_compile_options(tsr_perf_bench PRIVATE -w)
