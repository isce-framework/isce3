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
        Run geo2rdr.""")

    # Required arguments
    parser.add_argument('product', type=str,
        help='Input secondary HDF5 product.')
    parser.add_argument('topodir', type=str,
        help='Input topo directory.')

    # Optional arguments
    parser.add_argument('--azoff', type=float, action='store', default=0.0,
        help='Gross azimuth offset. Default: 0.0.')
    parser.add_argument('--rgoff', type=float, action='store', default=0.0,
        help='Gross range offset. Default: 0.0.')
    parser.add_argument('-f,--freq', action='store', type=str, default='A', dest='freq',
        help='Frequency code for product. Default: A.')
    parser.add_argument('-o,--output', type=str, action='store', default='offsets',
        help='Output directory. Default: offsets.', dest='outdir')

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

    # Create geo2rdr instance
    geo2rdrObj = isce3.geometry.geo2rdr(radarGrid=radarGrid,
                                    orbit=orbit,
                                    ellipsoid=ellipsoid)

    # Read topo multiband raster
    topoRaster = isce3.io.raster(filename=os.path.join(opts.topodir, 'topo.vrt'))

    # Init output directory
    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)

    # Run geo2rdr
    geo2rdrObj.geo2rdr(topoRaster, outputDir=opts.outdir, azshift=opts.azoff, rgshift=opts.rgoff)

if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
