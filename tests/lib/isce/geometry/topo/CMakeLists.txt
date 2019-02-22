foreach (TESTNAME topo)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    add_dependencies(${TESTNAME} geom_test_data)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} gtest)
    target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR})
    target_include_directories(${TESTNAME} PUBLIC 
        ${GDAL_INCLUDE_DIR}
        ${PYRE_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIR}
        ${ISCE_SOURCE_DIR}/contrib/cereal/include)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()

include(CTest)

# end of file
