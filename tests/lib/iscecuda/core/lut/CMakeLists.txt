add_isce_test(gpuLUT1d)
add_isce_test(gpuLUT2d)

file(COPY ${CMAKE_SOURCE_DIR}/tests/lib/isce/core/interpolator/data.txt
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
