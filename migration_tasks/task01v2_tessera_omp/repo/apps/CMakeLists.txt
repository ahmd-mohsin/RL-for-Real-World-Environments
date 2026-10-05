tsr_device_sources(tsr_render.cu)
add_executable(tsr_render tsr_render.cu)
target_link_libraries(tsr_render PRIVATE tessera)
tsr_target_warnings(tsr_render)
