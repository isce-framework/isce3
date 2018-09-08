
#Mimicking mm behaviour
set(PACKAGENAME product)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

#####Library headers
set(${PACKAGENAME}_EXPORT_HEADERS
    ComplexImagery.h
    Identification.h
    Identification.icc
    ImageMode.h
    ImageMode.icc
    Metadata.h
    Metadata.icc
    Product.h
    Product.icc
    Serialization.h
)

###Install headers into build/include
###This is where regex can be used on the header files if needed
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${${PACKAGENAME}_EXPORT_HEADERS})
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
