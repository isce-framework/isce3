if(NOSETESTS_FOUND)
    add_subdirectory(isce)
    message(STATUS "CUDA unit tests will be run for isce")
endif(NOSETESTS_FOUND)
