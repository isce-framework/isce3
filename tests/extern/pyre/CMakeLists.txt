add_test(import_pyre.py ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/import_pyre.py)
set_property(TEST import_pyre.py PROPERTY ENVIRONMENT
    PYTHONPATH=${CMAKE_INSTALL_PREFIX}/packages)
