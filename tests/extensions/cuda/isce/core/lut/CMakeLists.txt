add_isce_test(gpuLUT1d)
add_isce_test(gpuLUT2d)

file(COPY ../../../../../lib/isce/core/interpolator/data.txt
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
