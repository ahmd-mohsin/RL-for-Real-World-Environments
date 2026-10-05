# Hidden test suite for task01v2 (replaces tests/ at grading time).
# Requires the library target `tessera` with its offload flags as PUBLIC
# usage requirements.
set(TSR_HIDDEN_TESTS test_main.cpp)
file(GLOB TSR_HIDDEN_CASES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
list(REMOVE_ITEM TSR_HIDDEN_CASES ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)
add_executable(tsr_tests test_main.cpp ${TSR_HIDDEN_CASES})
target_link_libraries(tsr_tests PRIVATE tessera)
target_include_directories(tsr_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(tsr_tests PRIVATE -w)
target_compile_definitions(tsr_tests PRIVATE TSR_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data" TSR_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME tsr_tests COMMAND tsr_tests --no-intro --duration=false)

add_executable(tsr_offload_probe offload_probe.cpp)
target_link_libraries(tsr_offload_probe PRIVATE tessera ${CMAKE_DL_LIBS})
target_include_directories(tsr_offload_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(tsr_offload_probe PRIVATE -w)

add_executable(tsr_bench bench_tessera.cpp)
target_link_libraries(tsr_bench PRIVATE tessera)
target_include_directories(tsr_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(tsr_bench PRIVATE -w)
