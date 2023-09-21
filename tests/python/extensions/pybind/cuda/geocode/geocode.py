#/usr/bin/env python3

import itertools
import os

import numpy as np
from osgeo import gdal

import iscetest
import isce3
from nisar.products.readers import SLC

def test_cuda_geocode():
    rslc = SLC(hdf5file=os.path.join(iscetest.data, "envisat.h5"))

    dem_raster = isce3.io.Raster(os.path.join(iscetest.data,
                                             "geocode/zeroHeightDEM.geo"))

    # define geogrid
    epsg = 4326
    geogrid = isce3.product.GeoGridParameters(start_x=-115.65,
                                             start_y=34.84,
                                             spacing_x=0.0002,
                                             spacing_y=-8.0e-5,
                                             width=500,
                                             length=500,
                                             epsg=epsg)

    geotrans = [geogrid.start_x, geogrid.spacing_x, 0.0,
                     geogrid.start_y, 0.0, geogrid.spacing_y]

    # Init RadarGeometry, orbit, and doppler from RSLC
    radargrid = isce3.product.RadarGridParameters(os.path.join(iscetest.data,
                                                              "envisat.h5"))
    orbit = rslc.getOrbit()
    doppler = rslc.getDopplerCentroid()
    rdr_geometry = isce3.container.RadarGeometry(radargrid,
                                                orbit,
                                                doppler)

    # Set interp method
    interp_method = [isce3.core.DataInterpMethod.SINC]
    raster_dtype = [isce3.io.gdal.GDT_CFloat32]
    invalid_value = [np.nan]

    # If block mode, i.e. suffix != '', have lines per block smaller than
    # raster size. Otherwise set lines_per_block large enough to fit entire
    # raster.
    for xy, (suffix, lines_per_block) in itertools.product(['x', 'y'],
                                                          (['_blocked', 126],
                                                           ['', 1000])):
        # Init CUDA geocode obj
        cu_geocode = isce3.cuda.geocode.Geocode(
            geogrid, rdr_geometry, lines_per_block)

        # Put all input parameters in iterable list. geocode_rasters only
        # works with lists of the same length.
        output_raster = [isce3.io.Raster(f"{xy}{suffix}.geo", geogrid.width,
                                         geogrid.length, 1,
                                         gdal.GDT_CFloat32, "ENVI")]

        input_raster = [isce3.io.Raster(os.path.join(iscetest.data,
                                                     f"geocodeslc/{xy}.slc"))]

        cu_geocode.geocode_rasters(output_raster, input_raster, interp_method,
                                   raster_dtype, invalid_value, dem_raster)

        output_raster[0].set_geotransform(geotrans)

def test_validate():
    '''
    validate test outputs
    '''

    # check values of 2 geocoded rasters
    for xy, suffix in itertools.product(['x', 'y'], ['', '_blocked']):
        # get phase of complex test data and mask zeros
        test_raster = f"{xy + suffix}.geo"
        ds = gdal.Open(test_raster, gdal.GA_ReadOnly)
        test_arr = np.angle(ds.GetRasterBand(1).ReadAsArray())
        test_mask = test_arr == 0.0
        test_arr = np.ma.masked_array(test_arr, mask=test_mask)
        ds = None

        # get geotransform from test data
        geo_trans = isce3.io.Raster(test_raster).get_geotransform()
        x0 = np.radians(geo_trans[0] + geo_trans[1] / 2.0)
        dx = np.radians(geo_trans[1])
        y0 = np.radians(geo_trans[3] + geo_trans[5] / 2.0)
        dy = np.radians(geo_trans[5])

        # use geotransform to make lat/lon mesh
        pixels, lines = test_arr.shape
        meshx, meshy = np.meshgrid(np.arange(lines), np.arange(pixels))
        grid_lon = np.ma.masked_array(x0 + meshx * dx, mask=test_mask)
        grid_lat = np.ma.masked_array(y0 + meshy * dy, mask=test_mask)

        # calculate and check error within bounds
        if xy == 'x':
            err = np.nanmax(test_arr - grid_lon)
        else:
            err = np.nanmax(test_arr - grid_lat)
        assert(err < 1.0e-5), f'{test_raster} max error fail'

if __name__ == '__main__':
    test_cuda_geocode()
    test_validate()
