#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2020

import gdal
import numpy as np
import os
import iscetest

from nisar.workflows import geocode


class GeocodeOpts:
    '''
    class to emulate argparse terminal input
    member values set to test basic functionality
    values can be adjusted to meet test requirements
    '''
    raster = 'crossmul_cpu/crossmul.int'
    h5 = iscetest.data + 'envisat.h5'
    dem = iscetest.data + 'srtm_cropped.tif'
    outname = raster + '.geo'
    alks = rlks = 1


def testCpuGeocodeRun():
    '''
    run geocode on CPU
    '''
    # init inputs
    opts = GeocodeOpts()

    # run resamp
    geocode.main(opts)


def testCpuGeocodeMlookRun():
    '''
    run geocode on CPU
    '''
    # init inputs
    opts = GeocodeOpts()
    opts.alks = 3
    opts.rlks = 12

    # run resamp
    geocode.main(opts)


if __name__ == '__main__':
    testCpuGeocodeRun()
    testCpuGeocodeMlookRun()

# end of file
