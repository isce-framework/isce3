add_isce_test(lut1d)
add_isce_test(lut2d)

file(COPY ../interpolator/data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
