if(PYTEST_FOUND)
    add_subdirectory(isce)
    message(STATUS "python unit tests will be run for isce")
endif(PYTEST_FOUND)
