#Mimicking mm behaviour
set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

###Library to be built
set(${PACKAGENAME}_DLL isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

####Library sources
set(${PACKAGENAME}_SRCS 
              Baseline.cpp
              DateTime.cpp
              Doppler.cpp
              Ellipsoid.cpp
              EulerAngles.cpp
              Interpolator.cpp
              LUT2d.cpp
              LinAlg.cpp
              Metadata.cpp
              Orbit.cpp
              Peg.cpp
              Pegtrans.cpp
              Poly1d.cpp
              Poly2d.cpp
              Position.cpp
              Projections.cpp
              Quaternion.cpp
	      Raster.cpp
	      RasterLineIter.cpp		
)

#####Library headers
set(${PACKAGENAME}_EXPORT_HEADERS   Attitude.h 
                                    Baseline.h
                                    Constants.h
                                    DateTime.h
                                    Doppler.h
                                    Ellipsoid.h
                                    EulerAngles.h
                                    EulerAngles.icc
                                    Interpolator.h
                                    LUT2d.h
                                    LinAlg.h
                                    Metadata.h
                                    Orbit.h
                                    Peg.h
                                    Pegtrans.h
                                    Poly1d.h
                                    Poly2d.h
                                    Position.h
                                    Projections.h
                                    Quaternion.h
                                    Quaternion.icc
                                    Serialization.h
                                    Raster.h
				    Raster.icc
				    RasterLineIter.h
				    RasterLineIter.icc	
)

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
#Gdal is needed for Raster IO 
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

