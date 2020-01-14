set(TESTFILES
core/datetime_.py
core/timedelta.py
io/gdal/dataset.py
io/gdal/raster.py
)

foreach(TESTFILE ${TESTFILES})
    get_filename_component(NAMEWE ${TESTFILE} NAME_WE)
    get_filename_component(DIR ${TESTFILE} DIRECTORY)
    string(REPLACE "/" "." TESTNAME test.python.pybind_isce3.${DIR}.${NAMEWE})

    add_test(${TESTNAME} ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/${TESTFILE})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${PYTEST_LD_LIBRARY_PATH}
        PYTHONPATH=${PYTEST_PYTHONPATH})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
    set_property(TEST ${TESTNAME} PROPERTY WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
endforeach()
