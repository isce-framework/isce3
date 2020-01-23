#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import argparse
import os

import crossmul
import geo2rdr
import geocode
import rdr2geo
import resampSlc
import runargs

def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description="""
        Run interferogram.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('--yaml', type=str, required=True,
            help='Path the run parameter yaml file.')

    # Parse and return
    return parser.parse_args()

def main(path_yaml):
    args_dict = runargs.dictFromYaml(path_yaml)

    # run rdr2geo
    rdr2geo.main(runargs.Rdr2GeoArgs(args_dict))

    # run geo2rdr
    geo2rdr.main(runargs.Geo2RdrArgs(args_dict))

    # run resample secondary to reference
    resampSlc.main(runargs.ResampArgs(args_dict))

    # run crossmul
    crossmul_args = runargs.CrossmulArgs(args_dict)
    crossmul.main(crossmul_args)

    # geocode interferogram
    geocode_args = runargs.GeocodeArgs(args_dict)
    geocode_args.raster = crossmul_args.intFilePath
    geocode.main(geocode_args)

    # geocode coherence only if multilooked
    if (geocode_args.alks > 1 or geocode_args.rlks > 1):
        geocode_args.raster = crossmul_args.cohFilePath
        geocode.main(geocode_args)


if __name__ == '__main__':
    opts = cmdLineParse()
    main(opts.yaml)

# end of file
