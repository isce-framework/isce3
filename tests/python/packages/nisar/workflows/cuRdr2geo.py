#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import numpy as np
import os
import iscetest
from nisar.workflows import rdr2geo


class Rdr2GeoOpts:
    '''
    class to emulate argparse terminal input
    member values set to test basic functionality
    values can be adjusted to meet test requirements
    '''
    product = iscetest.data + 'envisat.h5'
    dem = iscetest.data + 'srtm_cropped.tif'
    freq = 'A'
    outdir = 'rdr2geo'
    mask = False

    gpu = True


def checkError(f_test, f_ref, dtype, tol, test_type):
    '''
    calculate error for file in vrt
    '''
    # retrieve data
    test = np.fromfile(f_test, dtype=dtype)
    ref = np.fromfile(f_ref, dtype=dtype)

    # calculate average error
    diff = np.abs(test - ref)
    diff = diff[diff < 5]
    error = np.mean(diff)

    # error check
    fname = os.path.basename(f_test)
    assert (error < tol), f'NISAR Python {test_type} rdr2geo fail at {fname}: {error} >= {tol}'


def testCudaRdr2Geo():
    '''
    run rdr2geo on GPU
    '''
    # init inputs
    opts = Rdr2GeoOpts()
    # unique cuda output for cuda geo2rdr to process from
    opts.outdir = 'rdr2geo_cuda'

    # run resamp
    rdr2geo.main(opts)

    # vrt constituent files to compare
    fnames = ['x.rdr', 'y.rdr', 'z.rdr', 'inc.rdr', 'hdg.rdr', 'localInc.rdr', 'localPsi.rdr']
    # dtypes of vrt constituent files
    dtypes = [np.float64, np.float64, np.float64, np.float32, np.float32, np.float32, np.float32]
    # tolerances per vrt constituent file
    tols = [1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02]

    # check errors
    for fname, dtype, tol in zip(fnames, dtypes, tols):
        checkError(os.path.join(opts.outdir, fname),
                os.path.join(iscetest.data, 'topo', fname),
                dtype,
                tol,
                'CUDA')



if __name__ == '__main__':
    testCudaRdr2Geo()

# end of file
