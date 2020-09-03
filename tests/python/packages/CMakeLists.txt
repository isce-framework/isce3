set(TESTFILES
isce3/core/ellipsoid.py
isce3/core/lut2d.py
isce3/core/orbit.py
isce3/core/testdatetime.py
isce3/geometry/geometry_equator.py
isce3/io/ih5py.py
isce3/io/raster.py
nisar/products/SLC.py
nisar/workflows/rdr2geo.py
nisar/workflows/geo2rdr.py
nisar/workflows/crossmul.py
nisar/workflows/resampSlc.py
nisar/workflows/geocode.py
pybind_nisar/workflows/focus.py
pybind_nisar/workflows/gcov.py
pybind_nisar/workflows/gslc.py
)

set(CUDA_TESTFILES
nisar/workflows/cuRdr2geo.py
nisar/workflows/cuGeo2rdr.py
nisar/workflows/cuCrossmul.py
nisar/workflows/cuResampSlc.py
)

if(WITH_CUDA)
    list(APPEND TESTFILES ${CUDA_TESTFILES})
endif()

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

set_tests_properties(test.python.pkg.nisar.workflows.geo2rdr PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.rdr2geo)
set_tests_properties(test.python.pkg.nisar.workflows.geocode PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.crossmul)
if(WITH_CUDA)
    set_tests_properties(test.python.pkg.nisar.workflows.cuGeo2rdr PROPERTIES
        DEPENDS test.python.pkg.nisar.workflows.cuRdr2geo)
endif()
