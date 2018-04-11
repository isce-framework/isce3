###Mimicking the structure used by mm
add_subdirectory(isce)

# Install dummy portinfo header file
file(WRITE ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/portinfo
     "// portinfo placeholder")
list(APPEND ${PACKAGENAME}_BUILD_HEADERS
     ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/portinfo)
