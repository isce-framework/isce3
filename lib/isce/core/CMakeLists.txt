#Mimicking mm behaviour
set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####Set the source files
set(LOCAL_SRCS
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
              ResampSlc.cpp
              TimeDelta.cpp)

####Append current path to every source file
set(${PACKAGENAME}_SRCS "")
foreach(CPPFILE ${LOCAL_SRCS})
    list(APPEND ${PACKAGENAME}_SRCS ${CMAKE_CURRENT_LIST_DIR}/${CPPFILE})
endforeach()

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
                                    ResampSlc.h
                                    ResampSlc.icc
                                    Serialization.h
                                    StateVector.h
                                    Tile.h
                                    Tile.icc
                                    TimeDelta.h)

###Install headers into build/include
###This is where regex can be used on the header files if needed
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${${PACKAGENAME}_EXPORT_HEADERS})
    message(STATUS "${HFILE} : to : ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE}")
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/${HFILE}
        ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE}
        COPYONLY
    )
    list(APPEND ${PACKAGENAME}_BUILD_HEADERS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE})
endforeach()

#Pyre stuff is needed for logging
#Gdal is needed for Raster IO 
include_directories(${${PACKAGENAME}_DLL} PUBLIC ${PYRE_INCLUDE_DIR} ${ISCE_BUILDINCLUDEDIR} ${GDAL_INCLUDE_DIR}) 

#Install headers as files
###This may be changed in future to install from build/include
install(FILES ${${PACKAGENAME}_BUILD_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}
    COMPONENT isce_headers)

# end of file
