foreach(TESTNAME geometry geometry_equator geometry_constlat)
    add_isce_test(${TESTNAME})
endforeach()

foreach(FILENAME input_data.txt output_data.txt output_data_zerodop.txt)
    file(COPY ${FILENAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
