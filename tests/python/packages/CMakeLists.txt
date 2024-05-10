set(TESTFILES
isce3/antenna/geometry_antenna.py
isce3/atmosphere/ionosphere.py
isce3/atmosphere/tec_products.py
isce3/cal/corner_reflector.py
isce3/cal/point_target_info.py
isce3/cal/radar_cross_section.py
isce3/core/gpu_check.py
isce3/core/llh.py
isce3/core/poly2d.py
isce3/core/resample_block_generators.py
isce3/geocode/geocode_slc.py
isce3/geometry/polygons.py
isce3/image/resample_slc.py
isce3/io/gdal/gdal_raster.py
isce3/io/background.py
isce3/matchtemplate/import_ampcor.py
isce3/matchtemplate/test_ampcor.py
isce3/signal/doppler_est_func.py
isce3/signal/fir_filter_func.py
isce3/signal/rfi_freq_null.py
isce3/signal/rfi_process_evd.py
isce3/unwrap/snaphu.py
nisar/antenna/beamformer.py
nisar/antenna/pattern.py
nisar/cal/corner_reflector_slc_func.py
nisar/cal/corner_reflector.py
nisar/cal/faraday_rotation_angle_slc.py
nisar/cal/pol_channel_imbalance_slc.py
nisar/mixed_mode/logic.py
nisar/mixed_mode/processing.py
nisar/pointing/doppler_lut_from_raw.py
#nisar/pointing/el_null_range_from_raw_ant.py
nisar/products/readers/antenna_parser.py
nisar/products/readers/attitude.py
nisar/products/readers/instrument_parser.py
nisar/products/readers/orbit.py
nisar/products/readers/raw.py
nisar/products/readers/rslc_cal.py
nisar/workflows/crossmul.py
nisar/workflows/estimate_abscal_factor.py
nisar/workflows/faraday_rot_angle_from_rslc.py
nisar/workflows/focus.py
nisar/workflows/gen_doppler_range_product.py
nisar/workflows/stage_dem.py
nisar/workflows/gcov.py
nisar/workflows/geo2rdr.py
nisar/workflows/geocode_insar.py
#nisar/workflows/gen_el_null_range_product.py
nisar/workflows/gen_el_rising_edge_product.py
nisar/workflows/gslc.py
nisar/workflows/helpers.py
nisar/workflows/insar.py
nisar/workflows/filter_interferogram.py
nisar/workflows/bandpass_insar.py
nisar/workflows/ionosphere.py
nisar/workflows/troposphere.py
nisar/workflows/baseline.py
nisar/workflows/process_args.py
nisar/workflows/rdr2geo.py
nisar/workflows/resample_slc.py
nisar/workflows/unwrap.py
nisar/workflows/point_target_analysis.py
nisar/workflows/noise_estimator.py
nisar/workflows/pol_channel_imbalance_from_rslc.py
)

if(WITH_CUDA)
    list(APPEND TESTFILES
            nisar/workflows/dense_offsets.py
            nisar/workflows/rubbersheet.py
            nisar/workflows/cuda_insar.py
        )
endif()

foreach(TESTFILE ${TESTFILES})
    get_filename_component(NAMEWE ${TESTFILE} NAME_WE)
    get_filename_component(DIR ${TESTFILE} DIRECTORY)
    string(REPLACE "/" "." TESTNAME test.python.pkg.${DIR}.${NAMEWE})

    add_test(${TESTNAME} ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/${TESTFILE})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${PYTEST_LD_LIBRARY_PATH}
        PYTHONPATH=${PYTEST_PYTHONPATH})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
    set_property(TEST ${TESTNAME} PROPERTY WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
endforeach()


set_tests_properties(test.python.pkg.nisar.workflows.geo2rdr PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.rdr2geo)
set_tests_properties(test.python.pkg.nisar.workflows.resample_slc PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.geo2rdr)
set_tests_properties(test.python.pkg.nisar.workflows.crossmul PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.resample_slc)
set_tests_properties(test.python.pkg.nisar.workflows.filter_interferogram PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.crossmul)

# using rdr2geo outputs as RUNW rasters to confirm geocode run
# using RUNW HDF5 needed as a verifiable dummy RUNW input
set_tests_properties(test.python.pkg.nisar.workflows.geocode_insar PROPERTIES
             DEPENDS "test.python.pkg.nisar.workflows.rdr2geo;test.python.pkg.nisar.workflows.unwrap")

set_tests_properties(test.python.pkg.nisar.workflows.insar PROPERTIES
             DEPENDS test.python.pkg.nisar.workflows.geocode_insar)

if(WITH_CUDA)
    set_tests_properties(test.python.pkg.nisar.workflows.dense_offsets PROPERTIES
                 DEPENDS test.python.pkg.nisar.workflows.cuda_insar)
    set_tests_properties(test.python.pkg.nisar.workflows.rubbersheet PROPERTIES
                 DEPENDS test.python.pkg.nisar.workflows.cuda_insar)
endif()
