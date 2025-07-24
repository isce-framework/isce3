include(Sources.cmake)
include(Headers.cmake)

configure_file(config.h.in config.h @ONLY)
configure_file(config.cpp.in config.cpp @ONLY)
list(APPEND SRCS ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
add_library(${LISCE} SHARED ${SRCS})

target_link_libraries(${LISCE} PUBLIC
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
    cxx_std_17
    )

target_include_directories(${LISCE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
    $<INSTALL_INTERFACE:"${ISCE_INCLUDEDIR}">
    )

# Define the preprocessor macro "ISCE3_CUDA" if CUDA is enabled
if(WITH_CUDA)
    target_compile_definitions(${LISCE} PUBLIC ISCE3_CUDA)
    add_subdirectory(cuda)
endif()

# Install public headers
foreach(HEADER ${HEADERS})
    get_filename_component(DIR ${HEADER} DIRECTORY)
    install(FILES       ${HEADER}
            DESTINATION "${ISCE_INCLUDEDIR}/isce3/${DIR}"
            COMPONENT   isce_libs)
endforeach()
install(FILES       ${CMAKE_CURRENT_BINARY_DIR}/config.h
        DESTINATION "${ISCE_INCLUDEDIR}/isce3/"
        COMPONENT   isce_libs)

# Specify API version and build version (used to generate name and soname
# symlinks on Unix systems).
set_target_properties(${LISCE} PROPERTIES
    SOVERSION   ${PROJECT_VERSION_MAJOR}
    VERSION     ${PROJECT_VERSION}
    )

# Install shared library
install(TARGETS ${LISCE}
        EXPORT isce3-targets
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
