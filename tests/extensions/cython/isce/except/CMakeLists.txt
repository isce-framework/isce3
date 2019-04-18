foreach (TESTNAME raster.py)
    configure_file(${TESTNAME} ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME})
    add_pytest(${TESTNAME})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${PYTEST_LD_LIBRARY_PATH}
        PYTHONPATH=${PYTEST_PYTHONPATH})
endforeach ()
