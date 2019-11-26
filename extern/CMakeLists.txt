add_subdirectory(googletest)
add_subdirectory(cereal)
if(NOT HAVE_PYRE)
    add_subdirectory(pyre)
endif()

add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM
    INTERFACE ${CMAKE_CURRENT_LIST_DIR}/eigen3)
