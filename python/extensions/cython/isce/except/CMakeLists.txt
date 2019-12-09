# Create library to hold exception handling stubs
add_library(cyerror STATIC "CyError.cpp")
set_property(TARGET cyerror PROPERTY POSITION_INDEPENDENT_CODE ON)

# Must link to ISCE error objects
message("${ISCE_BUILDINCLUDEDIR}")
target_include_directories(cyerror PRIVATE ${ISCE_BUILDINCLUDEDIR})

# Also, the cython extension needs access to the stub definitions
target_include_directories(isceextension PUBLIC ${CMAKE_CURRENT_LIST_DIR})
