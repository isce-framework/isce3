#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-
import os
from osgeo import gdal
import argparse
import warnings

import isce3
from nisar.products.readers import SLC

def cmdLineParser():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
            Run resampSlc.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                            
    # Required arguments
    parser.add_argument('--reference', type=str, required=True,
            help='File path for reference HDF5 product.')
    parser.add_argument('--secondary', type=str, required=True,
            help='File path for secondary HDF5 product that may contain a resampled SLC raster.')
    parser.add_argument('--frequency', type=str, required=True,
            help='Frequency of SLC.')
    parser.add_argument('--polarization', type=str, required=True,
            help='Polarization of SLC.')

    # Optional arguments
    parser.add_argument('--secondaryRaster', type=str, action='store', default='',
            help='Path to resampled SLC raster file. Use if resampled raster not in HDF5.')
    parser.add_argument('--azband', type=float, action='store', default=0.0,
            help='Azimuth bandwidth for azimuth commonband filtering.')
    parser.add_argument('--rgoff', type=str, action='store', default='',
            help='Filename for range offset raster for range commonbad filtering.')
    parser.add_argument('--alks', type=int, action='store', default=1,
            help='Number of looks to apply in the azimuth direction.')
    parser.add_argument('--rlks', type=int, action='store', default=1,
            help='Number of looks to apply in the range direction.')
    parser.add_argument('--cohFilePath', type=str, action='store', default='crossmul/crossmul.coh',
            help='Coherence output directory and file name.')
    parser.add_argument('--intFilePath', type=str, action='store', default='crossmul/crossmul.int',
            help='Interferogram output directory and file name.')
    parser.add_argument('-g', '--gpu', action='store_true', default=False,
        help='Enable GPU processing.')

    # Parse and return
    return parser.parse_args()


def main(opts):
    """
    crossmul
    """
    # prepare input rasters
    referenceSlc = SLC(hdf5file=opts.reference)
    referenceSlcDataset = referenceSlc.getSlcDataset(opts.frequency, opts.polarization)
    referenceSlcRaster = isce3.io.raster(filename='', h5=referenceSlcDataset)
    secondarySlc = SLC(hdf5file=opts.secondary)
    if opts.secondaryRaster:
        secondarySlcRaster = isce3.io.raster(filename=opts.secondaryRaster)
    else:
        secondarySlcDataset = secondarySlc.getSlcDataset(opts.frequency, opts.polarization)
        secondarySlcRaster = isce3.io.raster(filename='', h5=secondarySlcDataset)

    # prepare mulitlooked interferogram dimensions
    referenceGrid = referenceSlc.getRadarGrid(opts.frequency)
    length = int(referenceGrid.length / opts.alks)
    width = int(referenceGrid.width / opts.rlks)

    # init output directory(s)
    getDir = lambda filepath : os.path.split(filepath)[0]
    os.makedirs(getDir(opts.intFilePath), exist_ok=True)
    os.makedirs(getDir(opts.cohFilePath), exist_ok=True)

    # prepare output rasters
    driver = gdal.GetDriverByName('ISCE')
    igramDataset = driver.Create(opts.intFilePath, width, length, 1, gdal.GDT_CFloat32)
    igramRaster = isce3.io.raster(filename='', dataset=igramDataset)
    # coherence only generated when multilooked enabled
    if (opts.alks > 1 or opts.rlks > 1):
        cohDataset = driver.Create(opts.cohFilePath, width, length, 1, gdal.GDT_Float32)
        cohRaster = isce3.io.raster(filename='', dataset=cohDataset)
    else:
        cohRaster = None

    # prepare optional rasters
    if opts.rgoff:
        rgOffRaster = isce3.io.raster(filename=opts.rgoff)
    else:
        rgOffRaster = None

    if opts.azband:
        dopReference = referenceSlc.getDopplerCentroid()
        dopSecondary = secondarySlc.getDopplerCentroid()
        prf = referenceSlc.getSwathMetadata(opts.frequency).nominalAcquisitionPRF
        azimuthBandwidth = opts.azband
    else:
        dopReference = dopSecondary = None
        prf = azimuthBandwidth = 0.0

    # instantiate crossmul object based on user input
    if 'cuda' not in dir(isce3) and opts.gpu:
        warnings.warn('CUDA crossmul not available. Switching to CPU crossmul')
        opts.gpu = False

    if opts.gpu:
        crossmul = isce3.cuda.signal.crossmul()
    else:
        crossmul = isce3.signal.crossmul()

    crossmul.crossmul(referenceSlcRaster, secondarySlcRaster, igramRaster, cohRaster,
                      rngOffset=rgOffRaster, refDoppler=dopReference, secDoppler=dopSecondary,
                      rangeLooks=opts.rlks, azimuthLooks=opts.alks, prf=prf,
                      azimuthBandwidth=azimuthBandwidth)


if __name__ == '__main__':
    opts = cmdLineParser()
    main(opts)

# end of file
