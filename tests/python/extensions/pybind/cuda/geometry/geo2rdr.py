#!/usr/bin/env python3

import os

from osgeo import gdal
import numpy as np

import iscetest
import isce3.ext.isce3 as isce
from nisar.products.readers import SLC

def test_run():
    '''
    check if geo2rdr runs
    '''
    # prepare Geo2Rdr init params
    h5_path = os.path.join(iscetest.data, "envisat.h5")

    radargrid = isce.product.RadarGridParameters(h5_path)

    slc = SLC(hdf5file=h5_path)
    orbit = slc.getOrbit()
    doppler = slc.getDopplerCentroid()

    ellipsoid = isce.core.Ellipsoid()

    # require geolocation accurate to one millionth of a pixel.
    tol_pixels = 1e-6
    tol_seconds = tol_pixels / radargrid.prf

    # init Geo2Rdr class
    geo2rdr_obj = isce.cuda.geometry.Geo2Rdr(radargrid, orbit,
            ellipsoid, doppler, threshold=tol_seconds, numiter=50)

    # load rdr2geo unit test output
    rdr2geo_raster = isce.io.Raster("topo.vrt")

    # run
    geo2rdr_obj.geo2rdr(rdr2geo_raster, ".")

    # list of test outputs
    test_outputs = ["range.off", "azimuth.off"]

    # check each generated raster
    for test_output in test_outputs:
        # load dataset and get array
        test_ds = gdal.Open(test_output, gdal.GA_ReadOnly)
        test_arr = test_ds.GetRasterBand(1).ReadAsArray()

        # mask bad values
        test_arr = np.ma.masked_array(test_arr, mask=np.abs(test_arr) > 999.0)

        # compute max error (in pixels)
        test_err = np.max(np.abs(test_arr))

        # Error may slightly exceed tolerance since Newton step size isn't a
        # perfect estimate of the error in the solution.
        assert(test_err < 2 * tol_pixels), f"{test_output} accumulated error fail"


if  __name__ == "__main__":
    test_run()
