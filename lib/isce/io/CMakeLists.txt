#Mimicking mm behaviour
set(PACKAGENAME io)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

####List the source files
set(LOCAL_SRCS 
    IH5.cpp
    Raster.cpp
    )

####Append current path to every source file
set(${PACKAGENAME}_SRCS "")
foreach(CPPFILE ${LOCAL_SRCS})
    list(APPEND ${PACKAGENAME}_SRCS ${CMAKE_CURRENT_LIST_DIR}/${CPPFILE})
endforeach()

#####Library headers
set(${PACKAGENAME}_EXPORT_HEADERS
    Constants.h
    IH5.h
    IH5.icc
    Raster.h
    Raster.icc
    Serialization.h
)

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

#Install headers as files
###This may be changed in future to install from build/include
install(FILES ${${PACKAGENAME}_BUILD_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}
    COMPONENT isce_headers)

# end of file
