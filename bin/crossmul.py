#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-
import os
from osgeo import gdal
import argparse
from nisar.products.readers import SLC
from isce3.signal.Crossmul import Crossmul
from isce3.io.Raster import Raster

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
            Run resampSlc.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                            
    # Required arguments
    parser.add_argument('master', type=str,
            help='Filename for master HDF5 product.')
    parser.add_argument('slave', type=str,
            help='Filename for co-registered slave HDF5 product.')
    parser.add_argument('frequency', type=str,
            help='Frequency of SLC.')
    parser.add_argument('polarization', type=str,
            help='Polarization of SLC.')

    # Optional arguments
    parser.add_argument('azband', type=float, action='store', default=0.0,
            help='Azimuth bandwidth for azimuth commonbad filtering.')
    parser.add_argument('rgoff', type=str, action='store', default='',
            help='Filename for range offset raster for range commonbad filtering.')
    parser.add_argument('alks', type=int, action='store', default=1,
            help='Number of looks to apply in the azimuth direction.')
    parser.add_argument('rlks', type=int, action='store', default=1,
            help='Number of looks to apply in the range direction.')
    parser.add_argument('cohPathAndPrefix', type=str, action='store', default='crossmul/crossmul.coh',
            help='Coherence output directory and file name.')
    parser.add_argument('intPathAndPrefix', type=str, action='store', default='crossmul/crossmul.int',
            help='Interferogram output directory and file name.')

    # Parse and return
    return parser.parse_args()

def initDir(pathAndFile):
    path, file = os.path.split(pathAndFile)
    if not os.path.isdir(path):
        os.makedirs(path)

def main(opts):
    """
    crossmul
    """
    # prepare input rasters
    masterSlc = SLC(hdf5file=opts.master)
    masterSlcDataset = masterSlc.getSlcDataset(opts.frequency, opts.polarization)
    masterSlcRaster = Raster('', h5=masterSlcDataset)
    slaveSlc = SLC(hdf5file=opts.slave)
    slaveSlcDataset = slaveSlc.getSlcDataset(opts.frequency, opts.polarization)
    slaveSlcRaster = Raster('', h5=slaveSlcDataset)

    # prepare mulitlooked interferogram dimensions
    masterGrid = masterSlc.getRadarGrid(opts.frequency)
    length = int(masterGrid.length / opts.alks)
    width = int(masterGrid.width / opts.rlks)

    # init output directory(s)
    initDir(opts.intPathAndPrefix)
    initDir(opts.cohPathAndPrefix)

    # prepare output rasters
    driver = gdal.GetDriverByName('ISCE')
    igramDataset = driver.Create(opts.intPathAndPrefix, width, length, 1, gdal.GDT_CFloat32)
    igramRaster = Raster('', dataset=igramDataset)
    cohDataset = driver.Create(opts.cohPathAndPrefix, width, length, 1, gdal.GDT_Float32)
    cohRaster = Raster('', dataset=cohDataset)

    # prepare optional rasters
    if opts.rgoff:
        rgOffRaster = Raster(opts.rgoff)
    else:
        rgOffRaster = None

    if opts.azband:
        dopMaster = masterSlc.getDopplerCentroid()
        dopSlave = slaveSlc.getDopplerCentroid()
        prf = dopMaster.getRadarGrid(opts.frequency).prf
        azimuthBandwidth = opts.azband
    else:
        dopMaster = dopSlave = None
        prf = azimuthBandwidth = 0.0

    crossmul = Crossmul()
    crossmul.crossmul(masterSlcRaster, slaveSlcRaster, igramRaster, cohRaster,
                      rngOffset=rgOffRaster, refDoppler=dopMaster, secDoppler=dopSlave,
                      rangeLooks=opts.rlks, azimuthLooks=opts.alks, prf=prf,
                      azimuthBandwidth=azimuthBandwidth)


if __name__ == 'main':
    opts = cmdLineParser()
    main(opts)

# end of file
