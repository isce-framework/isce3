set(buildpkg ${CMAKE_BINARY_DIR}/packages)

file(GLOB_RECURSE pys RELATIVE ${CMAKE_CURRENT_LIST_DIR} pyre/*.py journal/*.py)
foreach(py ${pys})
    configure_file(${py} ${buildpkg}/${py} COPYONLY)
endforeach()

execute_process(COMMAND "date -u" OUTPUT_VARIABLE TODAY)
configure_file(pyre/meta ${buildpkg}/pyre/meta.py)
install(FILES ${buildpkg}/pyre/meta.py DESTINATION packages/pyre/)

install(DIRECTORY pyre journal
        DESTINATION packages
        FILES_MATCHING PATTERN *.py)
