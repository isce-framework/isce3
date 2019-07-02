# TODO: pass -Xcompiler=--coverage to nvcc, but only
#       when CUDA_HOST_COMPILER == CXX_COMPILER
set(COVERAGE_COMPILE_OPTS -O0 --coverage)
set(COVERAGE_LINK_OPTS --coverage)

function(SetCoverageOptions target)
    target_compile_options(${target} PRIVATE ${COVERAGE_COMPILE_OPTS})
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
        target_link_options  (${target} PRIVATE ${COVERAGE_LINK_OPTS})
    else()
        target_link_libraries(${target} PRIVATE ${COVERAGE_LINK_OPTS})
    endif()
endfunction()

# 'coverage' target to automatically run lcov with correct paths
add_custom_target(coverage)
# where to put the generated lcov info file
set(LCOV_INFO_FILE ${CMAKE_BINARY_DIR}/lcov.info)
add_custom_command(TARGET coverage POST_BUILD
    # Generate lcov info file
    COMMAND lcov --output-file ${LCOV_INFO_FILE}
                 --capture --directory ${CMAKE_BINARY_DIR}
    # Only show coverage for source files in this project
    COMMAND lcov --output-file ${LCOV_INFO_FILE}
                 --extract ${LCOV_INFO_FILE}
                   '${CMAKE_SOURCE_DIR}/lib/*'
                   '${CMAKE_SOURCE_DIR}/extensions/isce/cuda/*')
