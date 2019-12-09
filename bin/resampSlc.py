#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import os
import gdal
import argparse
from nisar.products.readers import SLC
from isce3.image.ResampSlc import ResampSlc
from isce3.io.Raster import Raster

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run resampSlc.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('product', type=str,
            help='Input HDF5 to be resampled.')
    parser.add_argument('frequency', type=str,
            help='Frequency of SLC.')
    parser.add_argument('polarization', type=str,
            help='Polarization of SLC.')

    # Optional arguments
    parser.add_argument('--outPathAndFile', type=str, action='store', default='resampSlc/resampSlc.slc',
            help='Output directory and file name.')
    parser.add_argument('--offsetdir', type=str, action='store', default='offsets',
            help='Input offset directory.')
    parser.add_argument('--linesPerTile', type=int, action='store', default=0,
            help='Number of lines resampled per iteration.')

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

    # instantiate resamp object
    resamp = ResampSlc(productGrid,
            productSlc.getDopplerCentroid(),
            productGrid.wavelength)
    
    # set number of lines per tile if arg > 0
    if opts.linesPerTile:
        resamp.linesPerTile = opts.linesPerTile

    # Prepare input rasters
    inSlcDataset = productSlc.getSlcDataset(opts.frequency, opts.polarization)
    inSlcRaster = Raster('', h5=inSlcDataset)
    azOffsetRaster = Raster(filename=os.path.join(opts.offsetdir, 'azimuth.off'))
    rgOffsetRaster = Raster(filename=os.path.join(opts.offsetdir, 'range.off'))

    # Init output directory
    if opts.outPathAndFile:
        path, file = os.path.split(opts.outPathAndFile)
        if not os.path.isdir(path):
            os.makedirs(path)

    # Prepare output raster
    driver = gdal.GetDriverByName('ISCE')
    slcPathAndName = opts.outPathAndFile
    outds = driver.Create(os.path.join(slcPathAndName), rgOffsetRaster.width,
            rgOffsetRaster.length, 1, gdal.GDT_CFloat32)
    outSlcRaster = Raster('', dataset=outds)

    # Run resamp
    resamp.resamp(inSlc=inSlcRaster,
            outSlc=outSlcRaster,
            rgoffRaster=rgOffsetRaster,
            azoffRaster=azOffsetRaster)


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
