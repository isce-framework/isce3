#Mimicking mm behaviour
set(PACKAGENAME dataio)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

###Library to be built
set(${PACKAGENAME}_DLL dataio.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

####Library sources
set(${PACKAGENAME}_SRCS 
              Raster.cpp
              RasterLineIter.cpp
              RasterBlockIter.cpp)

#####Library headers
set(${PACKAGENAME}_EXPORT_HEADERS   Raster.h
				    Raster.icc
                                    RasterLineIter.h
				    RasterLineIter.icc
                                    RasterBlockIter.h
				    RasterBlockIter.icc)

###Install headers into build/include
###This is where regex can be used on the header files if needed
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${${PACKAGENAME}_EXPORT_HEADERS})
    message(STATUS "${HFILE} : to : ${ISCE_BUILDINCLUDEDIR}/${PACKAGENAME}/${LOCALPROJ}/${HFILE}")
    configure_file(${HFILE} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE} COPYONLY)
    list(APPEND ${PACKAGENAME}_BUILD_HEADERS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE})
endforeach()

###Link sources to library
add_library(${${PACKAGENAME}_DLL} SHARED
                               ${${PACKAGENAME}_SRCS})


#Pyre stuff is needed for logging
target_include_directories(${${PACKAGENAME}_DLL} PUBLIC ${PYRE_INCLUDE_DIR} ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR}) 

# Add library dependencies 
target_link_libraries(${${PACKAGENAME}_DLL} ${GDAL_LIBRARY})

#Install headers as files
###This may be changed in future to install from build/include
install(FILES ${${PACKAGENAME}_BUILD_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}
    COMPONENT isce_headers)

#Install shared library
install(TARGETS ${${PACKAGENAME}_DLL}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_libs)

