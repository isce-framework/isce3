# Python packages
set(PYPKGS
    isce3
    pybind_nisar
    )

foreach(PYPKG ${PYPKGS})
    # Copy package directory to the binary directory for testing.
    file(COPY ${PYPKG} DESTINATION ${ISCE_BUILDPACKAGESDIR})

    # Install package directory.
    install(DIRECTORY ${PYPKG} DESTINATION ${ISCE_PACKAGESDIR})
endforeach()

##symlink defaults and schemas from install/share to pybind nisar workflows
set(WORKFLOWS
  defaults
  schemas
)
foreach(WORKFLOW ${WORKFLOWS})
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/share/nisar/${WORKFLOW}
            DESTINATION ${ISCE_PACKAGESDIR}/pybind_nisar/workflows)
endforeach()


# XXX Make workflow scripts executable.
set (list_of_exe
     pybind_nisar/workflows/crossmul.py
     pybind_nisar/workflows/focus.py
     pybind_nisar/workflows/geo2rdr.py
     pybind_nisar/workflows/geocode_insar.py
     pybind_nisar/workflows/gcov.py
     pybind_nisar/workflows/gslc.py
     pybind_nisar/workflows/insar.py
     pybind_nisar/workflows/rdr2geo.py
     pybind_nisar/workflows/resample_slc.py
     pybind_nisar/workflows/stage_dem.py
     pybind_nisar/workflows/unwrap.py)

foreach(script ${list_of_exe})
    install(
    PROGRAMS    ${script}
    DESTINATION ${ISCE_PACKAGESDIR}
    RENAME      ${script}
    )
endforeach()

