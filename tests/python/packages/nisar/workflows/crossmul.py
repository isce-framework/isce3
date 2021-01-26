#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import numpy as np
import iscetest
from nisar.workflows import crossmul

class workflowOpts:
    '''
    class to emulate argparse terminal input
    member values set to test basic functionality
    values can be adjusted to meet test requirements
    '''
    reference = secondary = iscetest.data + 'envisat.h5'
    secondaryRaster = ''
    frequency = 'A'
    polarization = 'HH'

    azband = 0.0
    rgoff = ''
    alks = 1
    rlks = 1
    intFilePath = 'crossmul_cpu/crossmul.int'
    cohFilePath = 'crossmul_cpu/crossmul.coh'

    gpu = False


def testCpuCrossmul():
    '''
    run crossmul SLC on CPU
    '''
    # init inputs
    opts = workflowOpts()

    # run resamp
    crossmul.main(opts)

    # check resulting interferogram has 0 phase
    igram = np.fromfile(opts.intFilePath, dtype=np.complex64)
    max_err = np.max(np.abs(np.angle(igram)))
    assert(max_err < 1e-6)


if __name__ == '__main__':
    testCpuCrossmul()

# end of file
