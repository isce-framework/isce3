include(Sources.cmake)
include(Headers.cmake)
install(FILES ${HEADERS} DESTINATION include/isce)

add_library(${LISCE} SHARED ${SRCS})
SetCoverageOptions(${LISCE})

target_link_libraries(${LISCE} PUBLIC
    Cereal::Cereal
    Eigen::Eigen
    ${GDAL_LIBRARY}
    ${HDF5_LIBRARY}
    OpenMP::OpenMP_CXX_Optional
    pyre
    FFTW::Float
    FFTW::Double
    FFTW::FloatThreads
    FFTW::DoubleThreads
    )

target_include_directories(${LISCE} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${GDAL_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIR}
    )

# Install shared library
install(TARGETS ${LISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
