include(Sources.cmake)
include(Headers.cmake)
install(FILES ${HEADERS} DESTINATION include/isce)

# Add OpenMP if found
if(OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(${LISCE} SHARED ${SRCS})
SetCoverageOptions(${LISCE})

target_link_libraries(${LISCE} PUBLIC
    cereal
    eigen
    ${GDAL_LIBRARY}
    ${HDF5_LIBRARY}
    pyre
    ${FFTWF_LIB} ${FFTW_LIB} ${FFTW_THREADS_LIB} ${FFTWF_THREADS_LIB})

target_include_directories(${LISCE} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${GDAL_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIR}
    )

# Install shared library
install(TARGETS ${LISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
