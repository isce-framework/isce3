#Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})

# Create a new ctest for TESTNAME.cpp
# Additional include directories can be specified after TESTNAME
function(add_isce_test TESTNAME)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} gtest)
    # If we're compiling against the CUDA ISCE library...
    if(DEFINED LISCECUDA)
        # Make CUDA libraries and headers available
        target_link_libraries(${TESTNAME} PUBLIC ${LISCECUDA})
        target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
        # CUDA doesn't support C++17 yet, so fall back to C++14
        #get_target_property(MYPROPS ${TESTNAME} COMPILE_OPTIONS)
        #string(REPLACE "c++17" "c++14" MYPROPS ${MYPROPS})
        #set_target_properties(${TESTNAME} PROPERTIES COMPILE_OPTIONS ${MYPROPS})
    endif()

    target_include_directories(${TESTNAME} PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${GDAL_INCLUDE_DIR}
        ${PYRE_INCLUDE_DIR}
        ${ISCE_SOURCE_DIR}/contrib/cereal/include
        gtest)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endfunction()

# Add a library subdirectory, containing source .cpp files and headers
function(add_isce_libdir PKGNAME CPPS HEADERS)
    string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

    # Prefix current path to each source file
    unset(SRCS)
    foreach(CPP ${CPPS})
        list(APPEND SRCS ${CMAKE_CURRENT_LIST_DIR}/${CPP})
    endforeach()

    # Add sources to library build requirements
    unset(SUBDIR)
    if(DEFINED LISCECUDA)
        set(SUBDIR "cuda")
        target_sources(${LISCECUDA} PRIVATE ${SRCS})
    else()
        target_sources(${LISCE} PRIVATE ${SRCS})
    endif()

    # Install headers to build/include
    # This is where regex can be used on headers if needed
    unset(BUILD_HEADERS)
    foreach(HFILE ${HEADERS})
        set(DEST "${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${SUBDIR}/${PKGNAME}/${HFILE}")
        configure_file(${CMAKE_CURRENT_LIST_DIR}/${HFILE} "${DEST}" COPYONLY)
        list(APPEND BUILD_HEADERS "${DEST}")
    endforeach()

    # Install headers as files
    # May be changed in the future to install from build/include
    install(FILES ${BUILD_HEADERS}
            DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${SUBDIR}/${PKGNAME}
            COMPONENT isce_headers)
endfunction()
