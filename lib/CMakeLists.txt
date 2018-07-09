###Mimicking the structure used by mm
add_subdirectory(isce)

# Install dummy portinfo header file
set(DUMMY_PORTINFO ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/portinfo)

if (EXISTS ${DUMMY_PORTINFO})
    message(STATUS "portinfo file alraedy appears to exist")
else()
    file(WRITE ${DUMMY_PORTINFO}
         "// portinfo placeholder")
endif()

list(APPEND ${PACKAGENAME}_BUILD_HEADERS
     ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/portinfo)
