set(TESTFILES
core/attitude.py
core/ellipsoid.py
core/poly1d.py
core/testdatetime.py
except/raster.py
geometry/deminterp.py
geometry/geometry.py
io/h5file.py
io/ih5py.py
io/raster.py
)

if(WITH_CUDA)
    list(APPEND TESTFILES cuda/cuimport/cuimport.py)
endif()

foreach(TESTFILE ${TESTFILES})
    get_filename_component(NAMEWE ${TESTFILE} NAME_WE)
    get_filename_component(DIR ${TESTFILE} DIRECTORY)
    string(REPLACE "/" "." TESTNAME test.python.cython.${DIR}.${NAMEWE})

    add_test(${TESTNAME} ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/${TESTFILE})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${PYTEST_LD_LIBRARY_PATH}
        PYTHONPATH=${PYTEST_PYTHONPATH})
endforeach()
