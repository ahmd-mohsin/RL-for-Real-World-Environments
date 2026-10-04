set(TSR_TEST_SOURCES
  test_main.cpp
  test_emu.cu
  test_prims.cu
  test_geom.cu
  test_bin.cu
  test_mesh.cu
  test_encode.cu
  test_raster.cu
  test_post_scene.cu
  test_render.cu
)

set(TSR_TEST_DEVICE_SOURCES ${TSR_TEST_SOURCES})
list(FILTER TSR_TEST_DEVICE_SOURCES INCLUDE REGEX "\\.cu$")
tsr_device_sources(${TSR_TEST_DEVICE_SOURCES})

add_executable(tsr_tests ${TSR_TEST_SOURCES})
target_link_libraries(tsr_tests PRIVATE tessera)
target_include_directories(tsr_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tsr_tests PRIVATE TSR_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
tsr_target_warnings(tsr_tests)

add_test(NAME tsr_tests COMMAND tsr_tests --no-intro --duration=false)
set_tests_properties(tsr_tests PROPERTIES ENVIRONMENT "TSR_EMU_WORKERS=4")
