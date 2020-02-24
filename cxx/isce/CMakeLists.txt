include(Sources.cmake)
include(Headers.cmake)

add_library(${LISCE} SHARED ${SRCS})
SetCoverageOptions(${LISCE})

target_link_libraries(${LISCE} PUBLIC
    Cereal::Cereal
    Eigen::Eigen
    ${GDAL_LIBRARY}
    ${HDF5_LIBRARY}
    pyre::pyre
    )


# These libraries are part of the interface but are not
# always needed. It's convenient to propagate them to other
# internal targets so we keep them in the build interface.
target_link_libraries(${LISCE} PUBLIC
    $<BUILD_INTERFACE:FFTW::Float>
    $<BUILD_INTERFACE:FFTW::Double>
    $<BUILD_INTERFACE:FFTW::FloatThreads>
    $<BUILD_INTERFACE:FFTW::DoubleThreads>
    )

target_link_libraries(${LISCE} PRIVATE
    OpenMP::OpenMP_CXX_Optional
    )

target_include_directories(${LISCE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
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
install(TARGETS ${LISCE} cereal eigen
        EXPORT isce3-targets
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
