include(IsceHelper)

# Create library target
add_library(${LISCE} SHARED)

add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(image)
add_subdirectory(io)
add_subdirectory(product)
add_subdirectory(radar)
add_subdirectory(signal)

# Add library dependencies
target_link_libraries(${LISCE} ${GDAL_LIBRARY} ${HDF5_LIBRARY} ${PYRE_LIBRARY}
    ${PYRE_JOURNAL_LIBRARY} ${FFTWF_LIB} ${FFTW_LIB} ${FFTW_THREADS_LIB} ${FFTWF_THREADS_LIB})

# Add include directories
# Pyre needed for journaling
include_directories(
    ${PYRE_INCLUDE_DIR}
    ${ISCE_BUILDINCLUDEDIR}
    ${GDAL_INCLUDE_DIR}
    ${CEREAL_SRC_INCLUDEDIR})

# Add OpenMP if found
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Install shared library
install(TARGETS ${LISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)
