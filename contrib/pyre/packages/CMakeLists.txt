set(buildpkg ${CMAKE_BINARY_DIR}/packages)

file(GLOB_RECURSE pys RELATIVE ${CMAKE_CURRENT_LIST_DIR} pyre/*.py journal/*.py)
foreach(py ${pys})
    configure_file(${py} ${buildpkg}/${py} COPYONLY)
endforeach()

execute_process(COMMAND "date -u" OUTPUT_VARIABLE TODAY)
configure_file(pyre/meta ${buildpkg}/pyre/meta.py)

install(DIRECTORY ${buildpkg} DESTINATION packages/)
