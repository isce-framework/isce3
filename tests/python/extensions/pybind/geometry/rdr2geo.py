#!/usr/bin/env python3

import itertools
import os
import types

from osgeo import gdal
import numpy as np
import pytest

import iscetest
import isce3
from isce3.geometry import (compute_east_north_ground_to_sat_vector,
                            Rdr2GeoParams)
from nisar.products.readers import SLC


def test_point():
    # Subset of tests/cxx/isce3/geometry/geometry/geometry.cpp
    fn = os.path.join(iscetest.data, "envisat.h5")
    slc = SLC(hdf5file=fn)
    orbit = slc.getOrbit()
    subband = "A"
    doplut = slc.getDopplerCentroid(frequency=subband)
    grid = slc.getRadarGrid(frequency=subband)

    # First row of input_data.txt
    dt = isce3.core.DateTime("2003-02-26T17:55:22.976222")
    r = 826988.6900674499
    h = 1777.0

    dem = isce3.geometry.DEMInterpolator(h)
    t = (dt - orbit.reference_epoch).total_seconds()
    dop = doplut.eval(t, r)
    wvl = grid.wavelength

    # native doppler, expect first row of output_data.txt
    llh = isce3.geometry.rdr2geo(t, r, orbit, grid.lookside, dop, wvl, dem)
    assert np.isclose(np.degrees(llh[0]), -115.44101120961082)
    assert np.isclose(np.degrees(llh[1]), 35.28794014757191)
    assert np.isclose(llh[2], 1777.0)

    # zero doppler, expect first row of output_data_zerodop.txt
    llh = isce3.geometry.rdr2geo(t, r, orbit, grid.lookside, 0.0, dem=dem)
    assert np.isclose(np.degrees(llh[0]), -115.43883834023249)
    assert np.isclose(np.degrees(llh[1]), 35.29610867314526)
    assert np.isclose(llh[2], 1776.9999999993)

    # try with bracketing method
    xyz = isce3.geometry.rdr2geo_bracket(t, r, orbit, grid.lookside, 0.0, 0.0,
                                         dem=dem)
    llh = isce3.core.Ellipsoid().xyz_to_lon_lat(xyz)
    assert np.isclose(np.degrees(llh[0]), -115.43883834023249)
    assert np.isclose(np.degrees(llh[1]), 35.29610867314526)
    assert np.isclose(llh[2], 1776.9999999993)

    # Run again without dem argument.  This is just to catch a former bug where
    # EPSG code was uninitialized.  Can't check answer since height is different
    # from default ctor.
    xyz = isce3.geometry.rdr2geo_bracket(t, r, orbit, grid.lookside, 0.0, 0.0)

    # Verify look side parsing
    with pytest.raises(ValueError):
        isce3.geometry.rdr2geo_bracket(t, r, orbit, "Foot", 0.0, 0.0)


@pytest.fixture(scope="module")
def unit_test_params():
    params = types.SimpleNamespace()

    # prepare Rdr2Geo init params
    h5_path = os.path.join(iscetest.data, "envisat.h5")

    radargrid = isce3.product.RadarGridParameters(h5_path)
    params.radargrid = radargrid

    params.slc = SLC(hdf5file=h5_path)
    orbit = params.slc.getOrbit()
    doppler = params.slc.getDopplerCentroid()

    ellipsoid = isce3.core.Ellipsoid()

    # init Rdr2Geo class
    params.rdr2geo_obj = isce3.geometry.Rdr2Geo(
        params.radargrid, orbit, ellipsoid, doppler
    )

    # load test DEM
    params.dem_raster = isce3.io.Raster(
        os.path.join(iscetest.data, "srtm_cropped.tif")
    )
    dem_interp = isce3.geometry.DEMInterpolator(params.dem_raster)
    dem_interp.compute_min_max_mean_height()

    # golden data layer tolerances
    params.tols = [1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02]

    # tolerances for east/north ground to satellite ENU unit vector layers
    # use same tolerance as heading angle since the computations are similar
    params.sat_to_ground_tols = [1.0e-4, 1.0e-4]

    return params


def test_run(unit_test_params):
    """
    check if topo runs with automatically generated layers
    """
    # run
    unit_test_params.rdr2geo_obj.topo(unit_test_params.dem_raster, ".")


def test_run_raster_layers(unit_test_params):
    """
    check if topo runs with manually set layers
    """
    radargrid = unit_test_params.radargrid
    length, width = radargrid.shape

    # list of tuples of file name w/o extension and gdal type
    fnames_dtypes = [
        ("x", gdal.GDT_Float64),
        ("y", gdal.GDT_Float64),
        ("z", gdal.GDT_Float64),
        ("inc", gdal.GDT_Float32),
        ("hdg", gdal.GDT_Float32),
        ("localInc", gdal.GDT_Float32),
        ("localPsi", gdal.GDT_Float32),
        ("simamp", gdal.GDT_Float32),
        ("layoverShadowMask", gdal.GDT_Byte),
        ("groundToSatEast", gdal.GDT_Float32),
        ("groundToSatNorth", gdal.GDT_Float32),
    ]

    # prepare test rasters
    [
        x_raster,
        y_raster,
        height_raster,
        incidence_angle_raster,
        heading_angle_raster,
        local_incidence_angle_raster,
        local_psi_raster,
        simulated_amplitude_raster,
        layover_shadow_raster,
        ground_to_sat_east_raster,
        ground_to_sat_north_raster,
    ] = [
        isce3.io.Raster(f"{fname}.rdr", width, length, 1, dtype, "ENVI")
        for fname, dtype in fnames_dtypes
    ]

    # run with prepared test rasters
    unit_test_params.rdr2geo_obj.topo(
        unit_test_params.dem_raster,
        x_raster,
        y_raster,
        height_raster,
        incidence_angle_raster,
        heading_angle_raster,
        local_incidence_angle_raster,
        local_psi_raster,
        simulated_amplitude_raster,
        layover_shadow_raster,
        ground_to_sat_east_raster,
        ground_to_sat_north_raster,
    )

    # combine all test rasters in a VRT
    _ = isce3.io.Raster(
        "topo_layers.vrt",
        raster_list=[
            x_raster,
            y_raster,
            height_raster,
            incidence_angle_raster,
            heading_angle_raster,
            local_incidence_angle_raster,
            local_psi_raster,
            simulated_amplitude_raster,
            ground_to_sat_east_raster,
            ground_to_sat_north_raster,
        ],
    )


def test_validate(unit_test_params):
    """
    validate generated results
    """
    # load reference topo raster
    ref_ds = gdal.Open(
        os.path.join(iscetest.data, "topo/topo.vrt"), gdal.GA_ReadOnly
    )
    golden_bands = range(1, ref_ds.RasterCount + 1)
    east_north_vec_bands = [8, 9]

    for test_path in ["topo.vrt", "topo_layers.vrt"]:
        # load generated topo raster
        test_ds = gdal.Open(test_path, gdal.GA_ReadOnly)

        # loop thru bands found in golden data and check tolerances
        for i_band, tol in zip(golden_bands, unit_test_params.tols):
            # retrieve test and ref arrays for current band
            test_arr = test_ds.GetRasterBand(i_band).ReadAsArray()
            ref_arr = ref_ds.GetRasterBand(i_band).ReadAsArray()

            # calculate mean of absolute error and mask anything > 5.0
            err = np.abs(test_arr - ref_arr)
            err = np.ma.masked_array(err, mask=err > 5.0)
            mean_err = np.mean(err)

            # check if tolerances met
            assert mean_err < tol, f"band {i_band} of {test_path} mean err fail"

        del test_ds
