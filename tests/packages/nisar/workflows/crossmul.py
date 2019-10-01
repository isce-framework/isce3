#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import numpy as np
from nisar.workflows import crossmul

class workflowOpts:
    '''
    class to emulate argparse terminal input
    member values set to test basic functionality
    values can be adjusted to meet test requirements
    '''
    master = slave = '../../../lib/isce/data/envisat.h5'
    frequency = 'A'
    polarization = 'HH'

    azband = 0.0
    rgoff = ''
    alks = 1
    rlks = 1
    intPathAndPrefix = 'crossmul/crossmul.int'
    cohPathAndPrefix = 'crossmul/crossmul.coh' 


def test_crossmul():
    '''
    run resample SLC without flattening and compare output against golden data
    '''
    # init inputs
    opts = workflowOpts()

    # run resamp
    crossmul.main(opts)

    # check resulting interferogram has 0 phase
    igram = np.fromfile('crossmul/crossmul.int', dtype=np.complex64)
    max_err = np.max(np.angle(igram))
    assert(max_err < 1e09)

if __name__ == '__main__':
    test_crossmul()

# end of file
