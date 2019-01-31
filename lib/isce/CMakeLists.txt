
# Pull in the source files
include(io/CMakeLists.txt)
include(radar/CMakeLists.txt)
include(product/CMakeLists.txt)
include(core/CMakeLists.txt)
include(image/CMakeLists.txt)
include(geometry/CMakeLists.txt)
include(signal/CMakeLists.txt)
include(unwrap/CMakeLists.txt)
# Set the output library
set(PACKAGENAME isce)
set(${PACKAGENAME}_DLL isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Link sources to library
add_library(${${PACKAGENAME}_DLL} SHARED
            ${io_SRCS}
            ${radar_SRCS}
            ${product_SRCS}
            ${core_SRCS}
            ${image_SRCS} 
            ${geometry_SRCS}
            ${signal_SRCS}
            ${unwrap/icu_SRCS})

# Add library dependencies 
target_link_libraries(${${PACKAGENAME}_DLL} ${GDAL_LIBRARY} ${HDF5_LIBRARY} ${PYRE_LIBRARY}
    ${PYRE_JOURNAL_LIBRARY} ${FFTWF_LIB} ${FFTW_LIB} ${FFTW_THREADS_LIB} ${FFTWF_THREADS_LIB})

# Add OpenMP if found
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Install shared library
install(TARGETS ${${PACKAGENAME}_DLL}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)

# end of file
