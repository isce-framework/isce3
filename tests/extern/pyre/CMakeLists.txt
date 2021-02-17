add_test(import_pyre.py ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/import_pyre.py)

set_property(TEST import_pyre.py PROPERTY ENVIRONMENT PYTHONPATH=${PYTEST_PYTHONPATH})
