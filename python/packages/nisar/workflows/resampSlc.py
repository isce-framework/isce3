#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import argparse
import gdal
import numpy as np
import os
import warnings

import isce3
from nisar.products.readers import SLC

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run resampSlc.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('--product', type=str, required=True,
            help='Input HDF5 to be resampled.')
    parser.add_argument('--frequency', type=str, required=True,
            help='Frequency of SLC.')
    parser.add_argument('--polarization', type=str, required=True,
            help='Polarization of SLC.')

    # Optional arguments
    parser.add_argument('--outFilePath', type=str, action='store', default='resampSlc/resampSlc.slc',
            help='Output directory and resampled raster file name.')
    parser.add_argument('--offsetdir', type=str, action='store', default='offsets',
            help='Input offset directory.')
    parser.add_argument('--linesPerTile', type=int, action='store', default=0,
            help='Number of lines resampled per iteration.')
    parser.add_argument('-g', '--gpu', action='store_true', default=False,
        help='Enable GPU processing.')

    # Parse and return
    return parser.parse_args()


def main(opts):
    """
    resample SLC
    """

    # prep SLC dataset input
    productSlc = SLC(hdf5file=opts.product)

    # get grids needed for resamp object instantiation
    productGrid = productSlc.getRadarGrid(opts.frequency)

    # instantiate resamp object based on user input
    if 'cuda' not in dir(isce3) and opts.gpu:
        warnings.warn('CUDA resamp not available. Switching to CPU resamp')
        opts.gpu = False

    if opts.gpu:
        resamp = isce3.cuda.image.resampSlc(radarGrid = productGrid,
                doppler = productSlc.getDopplerCentroid(),
                wavelength = productGrid.wavelength)
    else:
        resamp = isce3.image.resampSlc(radarGrid = productGrid,
                doppler = productSlc.getDopplerCentroid(),
                wavelength = productGrid.wavelength)
    
    # set number of lines per tile if arg > 0
    if opts.linesPerTile:
        resamp.linesPerTile = opts.linesPerTile

    # Prepare input rasters
    inSlcDataset = productSlc.getSlcDataset(opts.frequency, opts.polarization)
    inSlcRaster = isce3.io.raster(filename='', h5=inSlcDataset)
    azOffsetRaster = isce3.io.raster(filename=os.path.join(opts.offsetdir, 'azimuth.off'))
    rgOffsetRaster = isce3.io.raster(filename=os.path.join(opts.offsetdir, 'range.off'))

    # Init output directory
    if opts.outFilePath:
        path, _ = os.path.split(opts.outFilePath)
        os.makedirs(path, exist_ok=True)

    # Prepare output raster
    driver = gdal.GetDriverByName('ISCE')
    outds = driver.Create(opts.outFilePath, rgOffsetRaster.width,
            rgOffsetRaster.length, 1, gdal.GDT_CFloat32)
    outSlcRaster = isce3.io.raster(filename='', dataset=outds)

    # Run resamp
    resamp.resamp(inSlc=inSlcRaster,
            outSlc=outSlcRaster,
            rgoffRaster=rgOffsetRaster,
            azoffRaster=azOffsetRaster)


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
