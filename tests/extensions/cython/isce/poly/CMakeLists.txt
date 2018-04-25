foreach (TESTNAME poly1d.py)
    configure_file(${TESTNAME} ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME})
    add_nosetests(${TESTNAME})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${NOSE_LD_LIBRARY_PATH}
        PYTHONPATH=${NOSE_PYTHONPATH})
endforeach ()

include(CTest)
