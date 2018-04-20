#Mimicking mm behaviour
set(PACKAGENAME geometry)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####List the source files
set(LOCAL_SRCS 
    DEMInterpolator.cpp
    Geo2rdr.cpp
    Geometry.cpp
    Topo.cpp)

####Append current path to every source file
set(${PACKAGENAME}_SRCS "")
foreach(CPPFILE ${LOCAL_SRCS})
    list(APPEND ${PACKAGENAME}_SRCS ${CMAKE_CURRENT_LIST_DIR}/${CPPFILE})
endforeach()

#####Library headers
set(${PACKAGENAME}_EXPORT_HEADERS
    Basis.h
    DEMInterpolator.h
    Geo2rdr.h
    Geo2rdr.icc
    Geometry.h
    Pixel.h
    Serialization.h
    Topo.h
    Topo.icc
    TopoLayers.h)

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
