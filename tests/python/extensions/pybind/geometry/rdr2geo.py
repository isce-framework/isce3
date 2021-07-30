#!/usr/bin/env python3

import os
from osgeo import gdal
import numpy as np

import iscetest
import isce3
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
    h = 1777.

    dem = isce3.geometry.DEMInterpolator(h)
    t = (dt - orbit.reference_epoch).total_seconds()
    dop = doplut.eval(t, r)
    wvl = grid.wavelength

    # native doppler, expect first row of output_data.txt
    llh = isce3.geometry.rdr2geo(t, r, orbit, grid.lookside, dop, wvl, dem)
    assert np.isclose(np.degrees(llh[0]), -115.44101120961082)
    assert np.isclose(np.degrees(llh[1]), 35.28794014757191)
    assert np.isclose(llh[2], 1777.)

    # zero doppler, expect first row of output_data_zerodop.txt
    llh = isce3.geometry.rdr2geo(t, r, orbit, grid.lookside, 0.0, dem=dem)
    assert np.isclose(np.degrees(llh[0]), -115.43883834023249)
    assert np.isclose(np.degrees(llh[1]), 35.29610867314526)
    assert np.isclose(llh[2], 1776.9999999993)


def test_run():
    '''
    check if topo runs
    '''
    # prepare Rdr2Geo init params
    h5_path = os.path.join(iscetest.data, "envisat.h5")

    radargrid = isce3.product.RadarGridParameters(h5_path)

    slc = SLC(hdf5file=h5_path)
    orbit = slc.getOrbit()
    doppler = slc.getDopplerCentroid()

    ellipsoid = isce3.core.Ellipsoid()

    # init Rdr2Geo class
    rdr2geo_obj = isce3.geometry.Rdr2Geo(radargrid, orbit,
            ellipsoid, doppler)

    # load test DEM
    dem_raster = isce3.io.Raster(os.path.join(iscetest.data, "srtm_cropped.tif"))

    # run
    rdr2geo_obj.topo(dem_raster, ".")

def test_run_raster_layers():
    '''
    check if topo runs
    '''
    # prepare Rdr2Geo init params
    h5_path = os.path.join(iscetest.data, "envisat.h5")

    radargrid = isce3.product.RadarGridParameters(h5_path)

    slc = SLC(hdf5file=h5_path)
    orbit = slc.getOrbit()
    doppler = slc.getDopplerCentroid()

    ellipsoid = isce3.core.Ellipsoid()

    # init Rdr2Geo class
    rdr2geo_obj = isce3.geometry.Rdr2Geo(radargrid, orbit,
            ellipsoid, doppler)

    # load test DEM
    dem_raster = isce3.io.Raster(os.path.join(iscetest.data, "srtm_cropped.tif"))
    x_raster = isce3.io.Raster("x.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float64, 'ENVI')
    y_raster = isce3.io.Raster("y.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float64, 'ENVI')
    height_raster = isce3.io.Raster("z.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float64, 'ENVI')
    incidence_angle_raster = isce3.io.Raster("inc.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float32, 'ENVI')
    heading_angle_raster = isce3.io.Raster("hgd.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float32, 'ENVI')
    local_incidence_angle_raster = isce3.io.Raster("localInc.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float32, 'ENVI')
    local_Psi_raster = isce3.io.Raster("localPsi.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float32, 'ENVI')
    simulated_amplitude_raster = isce3.io.Raster("simamp.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float32, 'ENVI')
    shadow_layover_raster = isce3.io.Raster("mask.rdr", radargrid.width,
                                  radargrid.length, 1, gdal.GDT_Float32, 'ENVI')

    # run
    rdr2geo_obj.topo(dem_raster, x_raster,
                     y_raster, height_raster,
                     incidence_angle_raster,
                     heading_angle_raster,
                     local_incidence_angle_raster,
                     local_Psi_raster,
                     simulated_amplitude_raster, shadow_layover_raster)

    topo_raster = isce3.io.Raster(
            "topo_layers.vrt", raster_list=[x_raster,
                        y_raster, height_raster,
                        incidence_angle_raster,
                        heading_angle_raster,
                        local_incidence_angle_raster,
                        local_Psi_raster,
                        simulated_amplitude_raster])


def test_validate():
    '''
    validate generated results
    '''
    # load generated topo raster
    test_ds = gdal.Open("topo.vrt", gdal.GA_ReadOnly)

    # load reference topo raster
    ref_ds = gdal.Open(os.path.join(iscetest.data, "topo/topo.vrt"),
            gdal.GA_ReadOnly)

    # define tolerances
    tols = [1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02]

    # loop thru bands and check tolerances
    for i_band in range(ref_ds.RasterCount):
        # retrieve test and ref arrays for current band
        test_arr = test_ds.GetRasterBand(i_band+1).ReadAsArray()
        ref_arr = ref_ds.GetRasterBand(i_band+1).ReadAsArray()

        # calculate mean of absolute error and mask anything > 5.0
        err = np.abs(test_arr - ref_arr)
        err = np.ma.masked_array(err, mask=err > 5.0)
        mean_err = np.mean(err)

        # check if tolerances met
        assert( mean_err < tols[i_band]), f"band {i_band} mean err fail"

def test_layers_validate():
    '''
    validate generated results
    '''
    # load generated topo raster
    test_ds = gdal.Open("topo_layers.vrt", gdal.GA_ReadOnly)

    # load reference topo raster
    ref_ds = gdal.Open(os.path.join(iscetest.data, "topo/topo.vrt"),
            gdal.GA_ReadOnly)

    # define tolerances
    tols = [1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02]

    # loop thru bands and check tolerances
    for i_band in range(ref_ds.RasterCount):
        # retrieve test and ref arrays for current band
        test_arr = test_ds.GetRasterBand(i_band+1).ReadAsArray()
        ref_arr = ref_ds.GetRasterBand(i_band+1).ReadAsArray()

        # calculate mean of absolute error and mask anything > 5.0
        err = np.abs(test_arr - ref_arr)
        err = np.ma.masked_array(err, mask=err > 5.0)
        mean_err = np.mean(err)

        # check if tolerances met
        assert( mean_err < tols[i_band]), f"band {i_band} mean err fail"

