#!/usr/bin/env python3
  
import numpy as np
import argparse
import gdal
import sys
import os
import isce3
from nisar.products.readers import SLC

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run topo for master product.""")

    # Required arguments
    parser.add_argument('product', type=str,
        help='Input master HDF5 product.')
    parser.add_argument('dem', type=str,
        help='Input DEM raster (GDAL compatible).')

    # Optional arguments
    parser.add_argument('-o,--output', type=str, action='store', default='rdr2geo',
        help='Output directory. Default: topo.', dest='outdir')
    parser.add_argument('-f,--freq', action='store', type=str, default='A', dest='freq',
        help='Frequency code for product. Default: A.')
    parser.add_argument('-mask', action='store_true',
        help='Generate layover/shadow mask.')

    # Parse and return
    return parser.parse_args()

def main(opts):

    #instantiate slc object from NISAR SLC class
    slc = SLC(hdf5file=opts.product)
    
    # extract orbit
    orbit = slc.getOrbit()

    # extract the radar grid parameters 
    radarGrid = slc.getRadarGrid()

    # construct ellipsoid which is by default WGS84
    ellipsoid = isce3.core.ellipsoid()

    rdr2geo = isce3.geometry.rdr2geo(radarGrid=radarGrid,
                                    orbit=orbit, 
                                    ellipsoid=ellipsoid, 
                                    lookSide=slc.identification.lookDirection,
                                    computeMask=opts.mask)

    # Read DEM raster
    demRaster = isce3.io.raster(filename=opts.dem)

    # Init output directory
    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)

    # Run rdr2geo
    rdr2geo.topo(demRaster, outputDir=opts.outdir)

    return 0

if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
