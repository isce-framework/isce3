# Provide this location as a preprocessor macro
configure_file(testdata.h testdata.h)
add_compile_options(-include ${CMAKE_CURRENT_BINARY_DIR}/testdata.h)

include(isce/Sources.cmake)
list(TRANSFORM TESTFILES PREPEND isce/)

set(LIBS ${LISCE})

if(WITH_CUDA)
    include(iscecuda/Sources.cmake)
    list(TRANSFORM CUDA_TESTFILES PREPEND iscecuda/)

    # Get name of ISCE CUDA library
    include(IsceCuda)
    list(APPEND LIBS ${LISCECUDA})
    list(APPEND TESTFILES ${CUDA_TESTFILES})
endif()

foreach(FILENAME ${TESTFILES})
    get_filename_component(NAMEWE ${FILENAME} NAME_WE)
    get_filename_component(DIR ${FILENAME} DIRECTORY)
    string(REPLACE "/" "." TESTNAME test.cxx.${DIR}.${NAMEWE})
    set(TARGET ${NAMEWE})

    # FIXME hack in case of name collisions
    # otherwise all tests are dumped into ${CMAKE_CURRENT_BINARY_DIR}
    # tests shouldn't rely on existing in a certain directory
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})

    add_executable(${TARGET} ${FILENAME})
    add_test(NAME ${TESTNAME} COMMAND ${TARGET})

    # Change working directory so output data files won't be dumped in this
    # parent directory
    # TODO tests should specify their input/output data via absolute paths
    set_property(TEST ${TESTNAME} PROPERTY
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})

    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)

    target_link_libraries(${TARGET} PUBLIC ${LIBS} GTest::GTest)
endforeach()

set_tests_properties(test.cxx.isce.geometry.geo2rdr.geo2rdr PROPERTIES
             DEPENDS test.cxx.isce.geometry.topo.topo)
if(WITH_CUDA)
    set_tests_properties(test.cxx.iscecuda.geometry.geo2rdr.gpuGeo2rdr PROPERTIES
                 DEPENDS test.cxx.iscecuda.geometry.topo.gpuTopo)
endif()
