set(TESTFILES
pybind_nisar/products/readers/attitude.py
pybind_nisar/products/readers/orbit.py
pybind_nisar/workflows/crossmul.py
pybind_nisar/workflows/focus.py
pybind_nisar/workflows/stage_dem.py
pybind_nisar/workflows/gcov.py
pybind_nisar/workflows/geo2rdr.py
pybind_nisar/workflows/gslc.py
pybind_nisar/workflows/gpu_check.py
pybind_nisar/workflows/geocode_insar.py
pybind_nisar/workflows/insar.py
pybind_nisar/workflows/process_args.py
pybind_nisar/workflows/point_target_info.py
pybind_nisar/workflows/rdr2geo.py
pybind_nisar/workflows/resample_slc.py
pybind_nisar/workflows/unwrap.py
pybind_nisar/workflows/antenna_parser.py
)

if(WITH_CUDA)
    list(APPEND TESTFILES
            pybind_nisar/workflows/dense_offsets.py
            pybind_nisar/workflows/cuda_insar.py
        )
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

set_tests_properties(test.python.pkg.pybind_nisar.workflows.geo2rdr PROPERTIES
             DEPENDS test.python.pkg.pybind_nisar.workflows.rdr2geo)
set_tests_properties(test.python.pkg.pybind_nisar.workflows.resample_slc PROPERTIES
             DEPENDS test.python.pkg.pybind_nisar.workflows.geo2rdr)
if(WITH_CUDA)
    set_tests_properties(test.python.pkg.pybind_nisar.workflows.dense_offsets PROPERTIES
                 DEPENDS test.python.pkg.pybind_nisar.workflows.cuda_insar)
endif()
set_tests_properties(test.python.pkg.pybind_nisar.workflows.crossmul PROPERTIES
             DEPENDS test.python.pkg.pybind_nisar.workflows.resample_slc)
# using rdr2geo outputs as RUNW rasters to confirm geocode run
# using RUNW HDF5 needed as a verifiable dummy RUNW input
set_tests_properties(test.python.pkg.pybind_nisar.workflows.geocode_insar PROPERTIES
             DEPENDS "test.python.pkg.pybind_nisar.workflows.rdr2geo;test.python.pkg.pybind_nisar.workflows.unwrap")
set_tests_properties(test.python.pkg.pybind_nisar.workflows.insar PROPERTIES
             DEPENDS test.python.pkg.pybind_nisar.workflows.geocode_insar)
