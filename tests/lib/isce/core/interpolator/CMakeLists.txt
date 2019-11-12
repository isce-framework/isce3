add_isce_test(interpolator)
add_dependencies(interpolator intp_test_data)

add_custom_target(intp_test_data)
foreach (DATASRC data.txt)
    add_custom_command(TARGET intp_test_data PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E
                       copy  ${CMAKE_CURRENT_SOURCE_DIR}/${DATASRC} ${CMAKE_CURRENT_BINARY_DIR}/${DATASRC})
endforeach()

