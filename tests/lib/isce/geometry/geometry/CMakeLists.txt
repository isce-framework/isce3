foreach(TESTNAME geometry geometry_equator)
    add_isce_test(${TESTNAME})
    add_dependencies(${TESTNAME} geom_test_data)
endforeach()

foreach(FILENAME input_data.txt output_data.txt output_data_zerodop.txt)
    file(COPY ${FILENAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
