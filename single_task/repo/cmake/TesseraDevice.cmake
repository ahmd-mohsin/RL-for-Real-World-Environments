# Device sources (.cu) are compiled by nvcc when TSR_ENABLE_CUDA is ON and by
# the host C++ compiler against the emulated backend otherwise.
function(tsr_device_sources)
  if(TSR_ENABLE_CUDA)
    return()
  endif()
  foreach(src IN LISTS ARGN)
    # CMake passes "-x c++" for CXX sources with a non-C++ extension.
    set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
  endforeach()
endfunction()

function(tsr_target_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter>)
  endif()
endfunction()
