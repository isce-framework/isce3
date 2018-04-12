
# Pull in the source files
include(core/CMakeLists.txt)
include(geometry/CMakeLists.txt)

# Set the output library
set(PACKAGENAME isce)
set(${PACKAGENAME}_DLL isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

# Link sources to library
add_library(${${PACKAGENAME}_DLL} SHARED
            ${core_SRCS}
            ${geometry_SRCS})

# Add library dependencies 
target_link_libraries(${${PACKAGENAME}_DLL} ${GDAL_LIBRARY} ${PYRE_LIBRARY}
                      ${PYRE_JOURNAL_LIBRARY})

# Install shared library
install(TARGETS ${${PACKAGENAME}_DLL}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)

# end of file
