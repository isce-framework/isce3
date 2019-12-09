set(buildinclude ${CMAKE_BINARY_DIR}/include)
target_include_directories(pyre PUBLIC ${buildinclude})
target_link_options(pyre PUBLIC -static-libstdc++)

# Use CMake's configure_file variable substitution to set mm_platforms_*
# TODO more thorough platform detection
if (APPLE)
    set(PLATFORM darwin)
elseif(UNIX)
    set(PLATFORM linux)
endif()
configure_file(portinfo ${buildinclude}/portinfo)

add_subdirectory(pyre)
add_subdirectory(journal)
