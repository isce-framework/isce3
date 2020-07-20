#!/usr/bin/env python3

import os

from osgeo import gdal
import numpy as np

import iscetest
import pybind_isce3 as isce
from pybind_nisar.products.readers import SLC

def test_run():
    '''
    check if topo runs
    '''
    # prepare Rdr2Geo init params
    h5_path = os.path.join(iscetest.data, "envisat.h5")

    radargrid = isce.product.RadarGridParameters(h5_path)

    slc = SLC(hdf5file=h5_path)
    orbit = slc.getOrbit()
    doppler = slc.getDopplerCentroid()

    ellipsoid = isce.core.Ellipsoid()

    # init Rdr2Geo class
    rdr2geo_obj = isce.cuda.geometry.Rdr2Geo(radargrid, orbit,
            ellipsoid, doppler)

    # load test DEM
    dem_raster = isce.io.Raster(os.path.join(iscetest.data, "srtm_cropped.tif"))

    # run
    rdr2geo_obj.topo(dem_raster, ".")


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


if  __name__ == "__main__":
    test_run()
    test_validate()
