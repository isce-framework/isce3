set(TESTFILES
isce3/core/ellipsoid.py
isce3/core/orbit.py
isce3/core/testdatetime.py
isce3/geometry/geometry_equator.py
isce3/io/ih5py.py
isce3/io/raster.py
nisar/products/SLC.py
nisar/workflows/crossmul.py
nisar/workflows/resampSlc.py
)

foreach(TESTFILE ${TESTFILES})
    get_filename_component(NAMEWE ${TESTFILE} NAME_WE)
    get_filename_component(DIR ${TESTFILE} DIRECTORY)
    string(REPLACE "/" "." TESTNAME test.python.pkg.${DIR}.${NAMEWE})

    add_test(${TESTNAME} ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/${TESTFILE})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${PYTEST_LD_LIBRARY_PATH}
        PYTHONPATH=${PYTEST_PYTHONPATH})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
    set_property(TEST ${TESTNAME} PROPERTY WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
endforeach()

# FIXME this belongs with the rest of the test data
configure_file(nisar/products/swmr_test.h5 nisar/products COPYONLY)
