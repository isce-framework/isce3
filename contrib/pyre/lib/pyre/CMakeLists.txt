file(GLOB_RECURSE hfiles RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.h *.icc)
foreach(file ${hfiles})
    configure_file(${file} ${buildinclude}/pyre/${file} COPYONLY)
endforeach()

# XXX hack so that version.cc can include "version.h"
#     despite being moved to the build directory
configure_file(version.h    ${CMAKE_CURRENT_BINARY_DIR}/version.h)

configure_file(version      ${CMAKE_CURRENT_BINARY_DIR}/version.cc)
target_sources(pyre PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version.cc)

add_subdirectory(memory)
add_subdirectory(timers)
