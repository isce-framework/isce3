#Find pyre and journal libraries
find_library(LPYRE pyre HINTS ${PYRE_LIB_DIR})
find_library(LJOURNAL journal HINTS ${PYRE_LIB_DIR})

# Create a new ctest for TESTNAME.cpp
# Additional include directories can be specified after TESTNAME
function(add_isce_test TESTNAME)
    add_executable(${TESTNAME} ${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PUBLIC ${LISCE} ${LPYRE} ${LJOURNAL} gtest)
    # If we're compiling against the CUDA ISCE library...
    if(DEFINED LCUDAISCE)
        # Make CUDA libraries and headers available
        target_link_libraries(${TESTNAME} PUBLIC ${LCUDAISCE})
        target_include_directories(${TESTNAME} PUBLIC ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda)
        # CUDA doesn't support C++17 yet, so fall back to C++14
        get_target_property(MYPROPS ${TESTNAME} COMPILE_OPTIONS)
        string(REPLACE "c++17" "c++14" MYPROPS ${MYPROPS})
        set_target_properties(${TESTNAME} PROPERTIES COMPILE_OPTIONS ${MYPROPS})
    endif()

    target_include_directories(${TESTNAME} PUBLIC
        ${ISCE_BUILDINCLUDEDIR}
        ${GDAL_INCLUDE_DIR}
        ${PYRE_INCLUDE_DIR}
        ${ISCE_SOURCE_DIR}/contrib/cereal/include
        gtest)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endfunction()

