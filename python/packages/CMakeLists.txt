# Python packages
set(PYPKGS
    isce3
    nisar
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
            DESTINATION ${ISCE_PACKAGESDIR}/nisar/workflows)
endforeach()


# XXX Make workflow scripts executable.
set (list_of_exe
     nisar/workflows/crossmul.py
     nisar/workflows/estimate_abscal_factor.py
     nisar/workflows/faraday_rot_angle_from_rslc.py
     nisar/workflows/focus.py
     nisar/workflows/gen_el_rising_edge_product.py
     nisar/workflows/gen_el_null_range_product.py
     nisar/workflows/gen_doppler_range_product.py
     nisar/workflows/geo2rdr.py
     nisar/workflows/geocode_insar.py
     nisar/workflows/get_product_geometry.py
     nisar/workflows/gcov.py
     nisar/workflows/gslc.py
     nisar/workflows/gslc_point_target_analysis.py
     nisar/workflows/insar.py
     nisar/workflows/nisar_l0b_dm1_to_science.py
     nisar/workflows/noise_estimator.py
     nisar/workflows/point_target_analysis.py
     nisar/workflows/pol_channel_imbalance_from_rslc.py
     nisar/workflows/rdr2geo.py
     nisar/workflows/resample_slc.py
     nisar/workflows/stage_dem.py
     nisar/workflows/unwrap.py)

foreach(script ${list_of_exe})
    install(
    PROGRAMS    ${script}
    DESTINATION ${ISCE_PACKAGESDIR}
    RENAME      ${script}
    )
endforeach()

