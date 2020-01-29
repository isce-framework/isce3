add_custom_target(doxygen
    COMMAND doxygen ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
    )

set(sphxdir ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
add_custom_target(sphinx
    COMMAND sphinx-build -q -b html
        -c ${sphxdir}
        -d ${sphxdir}/_doctrees
        ${CMAKE_CURRENT_SOURCE_DIR}/sphinx
        ${sphxdir}/html
    )
