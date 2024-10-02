set(TESTFILES
config.py
core/attitude.py
core/basis.py
core/constants.py
core/datetime_.py
core/ellipsoid.py
core/euler.py
core/interp1d.py
core/interp2d.py
core/kernels.py
core/linspace.py
core/LUT1d.py
core/LUT2d.py
core/orbit.py
core/poly2d.py
core/projections.py
core/quaternion.py
core/statevector.py
core/timedelta.py
focus/backproject.py
focus/chirp.py
focus/presum.py
focus/rangecomp.py
geocode/geocodeCov.py
geocode/radar_grid_cube.py
geometry/bbox.py
geometry/dem.py
geometry/geo2rdr.py
geometry/rdr2geo.py
geometry/rtc.py
geometry/metadata_cubes.py
image/resamp.py
io/decode_bfpq_lut.py
io/gdal/dataset.py
io/gdal/raster.py
io/raster.py
math/stats.py
polsar/symmetrize.py
signal/convolve2D.py
signal/crossmul.py
signal/crossmultiply.py
signal/filter2D.py
signal/multilook.py
product/generic_product.py
product/geogridparameters.py
product/radargridparameters.py
product/sub_swaths.py
product/swath.py
product/grid.py
unwrap/icu.py
unwrap/phass.py
geometry/ltpcoordinates.py
geometry/pntintersect.py
geometry/look_inc_from_sr.py
antenna/frame.py
antenna/geometryfunc.py
antenna/el_pattern_est.py
antenna/edge_method_cost_func.py
#antenna/el_null_range_est.py
)

if(WITH_CUDA)
    list(APPEND TESTFILES
        cuda/core/compute_capability.py
        cuda/core/device.py
        cuda/focus/backproject.py
        cuda/geocode/geocode.py
        cuda/geometry/geo2rdr.py
        cuda/geometry/rdr2geo.py
        cuda/image/resamp.py
        cuda/signal/crossmul.py
        )
endif()

foreach(TESTFILE ${TESTFILES})
    get_filename_component(NAMEWE ${TESTFILE} NAME_WE)
    get_filename_component(DIR ${TESTFILE} DIRECTORY)
    string(REPLACE "/" "." TESTNAME test.python.pybind.${DIR}.${NAMEWE})

    add_test(${TESTNAME} ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/${TESTFILE})
    set_property(TEST ${TESTNAME} PROPERTY ENVIRONMENT
        LD_LIBRARY_PATH=${PYTEST_LD_LIBRARY_PATH}
        PYTHONPATH=${PYTEST_PYTHONPATH})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
    set_property(TEST ${TESTNAME} PROPERTY WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIR})
endforeach()

set_tests_properties(test.python.pybind.geometry.geo2rdr PROPERTIES
             DEPENDS test.python.pybind.geometry.rdr2geo)
if(WITH_CUDA)
    set_tests_properties(test.python.pybind.cuda.geometry.geo2rdr PROPERTIES
                 DEPENDS test.python.pybind.cuda.geometry.rdr2geo)
endif()
