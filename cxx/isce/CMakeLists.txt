include(Sources.cmake)
include(Headers.cmake)

add_library(${LISCE} SHARED ${SRCS})
SetCoverageOptions(${LISCE})

target_link_libraries(${LISCE} PUBLIC
    Cereal::Cereal
    Eigen3::Eigen
    GDAL::GDAL
    HDF5::HDF5
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
    project_warnings
    )

target_compile_features(${LISCE} INTERFACE
    cxx_std_14
    )

target_include_directories(${LISCE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
    )

# Define the preprocessor macro "ISCE3_CUDA" if CUDA is enabled
if(WITH_CUDA)
    target_compile_definitions(${LISCE} PUBLIC ISCE3_CUDA)
endif()

# Install public headers
foreach(HEADER ${HEADERS})
    get_filename_component(DIR ${HEADER} DIRECTORY)
    install(FILES       ${HEADER}
            DESTINATION include/isce/${DIR}
            COMPONENT   isce_libs)
endforeach()

# Install shared library
install(TARGETS ${LISCE}
        EXPORT isce3-targets
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
