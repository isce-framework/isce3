include(Sources.cmake)
include(Headers.cmake)

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

# Install public headers
foreach(HEADER ${HEADERS})
    get_filename_component(DIR ${HEADER} DIRECTORY)
    install(FILES       ${HEADER}
            DESTINATION include/isce/${DIR}
            COMPONENT   isce_libs)
endforeach()

# Install shared library
install(TARGETS ${LISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
