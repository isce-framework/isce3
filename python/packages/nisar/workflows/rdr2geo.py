#!/usr/bin/env python3
  
import argparse
import os
import warnings

import isce3
from nisar.products.readers import SLC

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run topo for reference product.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('-p', '--product', type=str, required=True,
        help='Input reference HDF5 product.')
    parser.add_argument('-d', '--dem', type=str, required=True,
        help='Input DEM raster (GDAL compatible).')

    # Optional arguments
    parser.add_argument('-o', '--output', type=str, action='store', default='rdr2geo',
        help='Output directory.', dest='outdir')
    parser.add_argument('-f', '--freq', action='store', type=str, default='A', dest='freq',
        help='Frequency code for product.')
    parser.add_argument('-m', '--mask', action='store_true',
        help='Generate layover/shadow mask.')
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

    # instantiate rdr2geo object based on user input
    if 'cuda' not in dir(isce3) and opts.gpu:
        warnings.warn('CUDA rdr2geo not available. Switching to CPU rdr2geo')
        opts.gpu = False

    if opts.gpu:
        rdr2geo = isce3.cuda.geometry.rdr2geo(radarGrid=radarGrid,
                                        orbit=orbit, 
                                        ellipsoid=ellipsoid, 
                                        computeMask=opts.mask,
                                        doppler=doppler)
    else:
        rdr2geo = isce3.geometry.rdr2geo(radarGrid=radarGrid,
                                        orbit=orbit, 
                                        ellipsoid=ellipsoid, 
                                        computeMask=opts.mask,
                                        doppler=doppler)

    # Read DEM raster
    demRaster = isce3.io.raster(filename=opts.dem)

    # Init output directory
    os.makedirs(opts.outdir, exist_ok=True)

    # Run rdr2geo
    rdr2geo.topo(demRaster, outputDir=opts.outdir)

    return 0

if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts)

# end of file
