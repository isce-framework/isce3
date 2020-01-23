#!/usr/bin/env python3
  
import numpy as np
import argparse
import gdal
import sys
import os
import warnings

import isce3
from nisar.products.readers import SLC

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run geo2rdr.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('-p', '--product', type=str, required=True,
        help='Input HDF5 product.')

    # Optional arguments
    parser.add_argument('-t', '--topopath', type=str, action='store', default='rdr2geo/topo.vrt',
        help='Input topo file path.')
    parser.add_argument('--azoff', type=float, action='store', default=0.0,
        help='Gross azimuth offset.')
    parser.add_argument('--rgoff', type=float, action='store', default=0.0,
        help='Gross range offset.')
    parser.add_argument('-f', '--freq', action='store', type=str, default='A', dest='freq',
        help='Frequency code for product.')
    parser.add_argument('-o', '--output', type=str, action='store', default='offsets',
        help='Output directory.', dest='outdir')
    parser.add_argument('-g', '--gpu', action='store_true', default=False,
        help='Enable GPU processing.')

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

    # get doppler centroid
    doppler = slc.getDopplerCentroid()

    # instantiate geo2rdr object based on user input
    if 'cuda' not in dir(isce3) and opts.gpu:
        warnings.warn('CUDA geo2rdr not available. Switching to CPU geo2rdr')
        opts.gpu = False

    if opts.gpu:
        geo2rdrObj = isce3.cuda.geometry.geo2rdr(radarGrid=radarGrid,
                                            orbit=orbit,
                                            ellipsoid=ellipsoid,
                                            doppler=doppler,
                                            threshold=1e-9)
    else:
        geo2rdrObj = isce3.geometry.geo2rdr(radarGrid=radarGrid,
                                            orbit=orbit,
                                            ellipsoid=ellipsoid,
                                            doppler=doppler,
                                            threshold=1e-9)

    # Read topo multiband raster
    topoRaster = isce3.io.raster(filename=opts.topopath)

    # Init output directory
    os.makedirs(opts.outdir, exist_ok=True)

    # Run geo2rdr
    geo2rdrObj.geo2rdr(topoRaster, outputDir=opts.outdir, azshift=opts.azoff, rgshift=opts.rgoff)

if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
