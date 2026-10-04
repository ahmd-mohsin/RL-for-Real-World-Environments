set(TSR_TEST_SOURCES
  test_main.cpp
  test_emu.cu
  test_prims.cu
)

set(TSR_TEST_DEVICE_SOURCES ${TSR_TEST_SOURCES})
list(FILTER TSR_TEST_DEVICE_SOURCES INCLUDE REGEX "\\.cu$")
tsr_device_sources(${TSR_TEST_DEVICE_SOURCES})

add_executable(tsr_tests ${TSR_TEST_SOURCES})
target_link_libraries(tsr_tests PRIVATE tsr_prims)
target_include_directories(tsr_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest ${CMAKE_CURRENT_SOURCE_DIR})
tsr_target_warnings(tsr_tests)

add_test(NAME tsr_tests COMMAND tsr_tests --no-intro --duration=false)
set_tests_properties(tsr_tests PROPERTIES ENVIRONMENT "TSR_EMU_WORKERS=4")
