#Mimicking mm behaviour
set(PACKAGENAME signal)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####List the source files
set(LOCAL_SRCS 
    Signal.cpp
    Crossmul.cpp
    Filter.cpp
    Looks.cpp)

####Append current path to every source file
set(${PACKAGENAME}_SRCS "")
foreach(CPPFILE ${LOCAL_SRCS})
    list(APPEND ${PACKAGENAME}_SRCS ${CMAKE_CURRENT_LIST_DIR}/${CPPFILE})
endforeach()

#####Library headers
set(${PACKAGENAME}_EXPORT_HEADERS
    Signal.h
    Signal.icc
    fftw3cxx.h
    Crossmul.h
    Crossmul.icc
    Filter.h
    Looks.h
    Looks.icc)

###Install headers into build/include
###This is where regex can be used on the header files if needed
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${${PACKAGENAME}_EXPORT_HEADERS})
#    message(STATUS "${HFILE} : to : ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE}")
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/${HFILE}
        ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE}
        COPYONLY
    )
    list(APPEND ${PACKAGENAME}_BUILD_HEADERS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/${HFILE})
endforeach()

#Pyre stuff is needed for logging
#Gdal is needed for Raster IO 
include_directories(${${PACKAGENAME}_DLL}
    PUBLIC
    ${PYRE_INCLUDE_DIR}
    ${ISCE_BUILDINCLUDEDIR}
    ${GDAL_INCLUDE_DIR}
    ${CEREAL_SRC_INCLUDEDIR}
) 

#Install headers as files
###This may be changed in future to install from build/include
install(FILES ${${PACKAGENAME}_BUILD_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}
    COMPONENT isce_headers)

# end of file
