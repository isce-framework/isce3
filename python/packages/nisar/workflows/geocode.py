#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import argparse
import gdal
import numpy as np

import isce3
from isce3.io.Raster import Raster
from nisar.products.readers import SLC

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run interferogram.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('-r', '--raster', type=str, required=True,
            help='Path of input raster to geocode.')
    parser.add_argument('--h5', type=str, required=True,
        help='Path to associated HDF5 product.')
    parser.add_argument('-d', '--dem', type=str, required=True,
        help='Input DEM raster (GDAL compatible).')

    # Optional arguments
    parser.add_argument('--outname', type=str, action='store', default='',
            help='Name of geocoded output.')
    parser.add_argument('--alks', type=int, action='store', default=1,
            help='Number of looks to apply in the azimuth direction.')
    parser.add_argument('--rlks', type=int, action='store', default=1,
            help='Number of looks to apply in the range direction.')

    # Parse and return
    return parser.parse_args()

def main(opts):
    """
    Runs isce::geometry::Geocode for any GDAL raster with an associated HDF5 product.
    For example, to geocode a multilooked interferogram, provide the HDF5 product
    for the reference scene which defined the full-resolution radar geometry.
    """
    # Common driver for all output files
    driver = gdal.GetDriverByName('ISCE')

    # Open input raster
    input_raster = Raster(opts.raster)

    # Open its associated product
    slc = SLC(hdf5file=opts.h5)

    # Make ellipsoid
    ellps = isce3.core.ellipsoid()

    # Get radar grid
    radar_grid = slc.getRadarGrid()
    if (opts.alks > 1 or opts.rlks > 1):
        radar_grid = radar_grid.multilook(opts.alks, opts.rlks)

    # Get orbit
    orbit = slc.getOrbit()

    # Make reference epochs consistent
    orbit.referenceEpoch = radar_grid.refEpoch

    # Make a zero-Doppler LUT
    doppler = isce3.core.lut2d()

    # Compute DEM bounds for radar grid 
    proj_win = isce3.geometry.getBoundsOnGround(orbit, ellps, doppler, radar_grid.lookSide,
                                                radar_grid, 0, 0, radar_grid.width, radar_grid.length,
                                                margin=np.radians(0.01))
    # GDAL expects degrees
    proj_win = np.degrees(proj_win)

    # Extract virtual DEM covering radar bounds
    ds = gdal.Open(opts.dem, gdal.GA_ReadOnly)
    crop_dem_ds = gdal.Translate('/vsimem/dem.crop', ds, format='ISCE', projWin=proj_win)
    ds = None

    # Instantiate Geocode object
    geo = isce3.geometry.geocode(orbit=orbit,
            ellipsoid=ellps,
            inputRaster=input_raster)

    # Set radar grid
    geo.radarGrid(doppler,
		radar_grid.refEpoch,
		radar_grid.sensingStart,
		1.0/radar_grid.prf,
                radar_grid.length,
		radar_grid.startingRange,
		radar_grid.rangePixelSpacing,
                radar_grid.wavelength,
		radar_grid.width,
		radar_grid.lookSide)

    # Get DEM geotransform from DEM raster
    lon0, dlon, _, lat0, _, dlat = crop_dem_ds.GetGeoTransform()
    ny_geo = crop_dem_ds.RasterYSize
    nx_geo = crop_dem_ds.RasterXSize
    crop_dem_ds = None
    print('Cropped DEM shape: (%d, %d)' % (ny_geo, nx_geo))

    # Open DEM raster as an ISCE raster
    dem_raster = Raster('/vsimem/dem.crop')

    # Set geographic grid
    geo.geoGrid(lon0, lat0, dlon, dlat, nx_geo, ny_geo, dem_raster.EPSG)

    # Create output raster
    if opts.outname == '':
        opts.outname = opts.raster + '.geo'
    odset = driver.Create(opts.outname, nx_geo, ny_geo, 1, input_raster.getDatatype(band=1))
    output_raster = Raster('', dataset=odset)

    # Run geocoding
    geo.geocode(input_raster, output_raster, dem_raster)

    # Clean up
    crop_dem_ds = None
    odset = None
    gdal.Unlink('/vsimem/dem.crop')

if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
