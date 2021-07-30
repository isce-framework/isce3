#!/usr/bin/env python3
import os
import numpy as np
from osgeo import gdal
import iscetest
import pybind_isce3 as isce
from nisar.products.readers import SLC

def test_run():
    '''
    run geocodeSlc bindings with same parameters as C++ test to make sure it does not crash
    '''
    # load h5 for doppler and orbit
    rslc = SLC(hdf5file=os.path.join(iscetest.data, "envisat.h5"))

    # define geogrid
    geogrid = isce.product.GeoGridParameters(start_x=-115.65,
        start_y=34.84,
        spacing_x=0.0002,
        spacing_y=-8.0e-5,
        width=500,
        length=500,
        epsg=4326)

    # define geotransform
    geotrans = [geogrid.start_x, geogrid.spacing_x, 0.0,
            geogrid.start_y, 0.0, geogrid.spacing_y]

    img_doppler = rslc.getDopplerCentroid()
    native_doppler = isce.core.LUT2d(img_doppler.x_start,
            img_doppler.y_start, img_doppler.x_spacing,
            img_doppler.y_spacing, np.zeros((geogrid.length,geogrid.width)))

    dem_raster = isce.io.Raster(os.path.join(iscetest.data, "geocode/zeroHeightDEM.geo"))

    radargrid = isce.product.RadarGridParameters(os.path.join(iscetest.data, "envisat.h5"))

    # geocode same 2 rasters as C++ version
    for xy in ['x', 'y']:
        out_raster = isce.io.Raster(f"{xy}.geo", geogrid.width, geogrid.length, 1,
                gdal.GDT_CFloat32, "ENVI")

        in_raster = isce.io.Raster(os.path.join(iscetest.data, f"geocodeslc/{xy}.slc"))

        isce.geocode.geocode_slc(output_raster=out_raster,
            input_raster=in_raster,
            dem_raster=dem_raster,
            radargrid=radargrid,
            geogrid=geogrid,
            orbit=rslc.getOrbit(),
            native_doppler=native_doppler,
            image_grid_doppler=img_doppler,
            ellipsoid=isce.core.Ellipsoid(),
            threshold_geo2rdr=1.0e-9,
            numiter_geo2rdr=25,
            lines_per_block=1000,
            dem_block_margin=0.1,
            flatten=False)

        out_raster.set_geotransform(geotrans)

def test_validate():
    '''
    validate test outputs
    '''

    # check values of 2 geocoded rasters
    for xy in ['x', 'y']:
        # get phase of complex test data and mask zeros
        test_raster = f"{xy}.geo"
        ds = gdal.Open(test_raster, gdal.GA_ReadOnly)
        test_arr = np.angle(ds.GetRasterBand(1).ReadAsArray())
        test_mask = test_arr == 0.0
        test_arr = np.ma.masked_array(test_arr, mask=test_mask)
        ds = None

        # get geotransform from test data
        geo_trans = isce.io.Raster(test_raster).get_geotransform()
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


if __name__ == "__main__":
    test_run()
    test_validate()
