foreach(TESTNAME geometry geometry_equator geometry_constlat)
    add_isce_test(${TESTNAME})
    add_dependencies(${TESTNAME} geom_test_data)
    add_dependencies(${TESTNAME} geom_text_data)
endforeach()

add_custom_target(geom_text_data)
foreach(FILENAME input_data.txt output_data.txt output_data_zerodop.txt)
    add_custom_command(TARGET geom_text_data PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E 
                       copy ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME})
endforeach()
