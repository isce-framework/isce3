#!/usr/bin/env python3
#
# Author: Liang Yu
# Copyright 2019-

import numpy as np
import os
import iscetest
from nisar.workflows import geo2rdr


class Geo2RdrOpts:
    '''
    class to emulate argparse terminal input
    member values set to test basic functionality
    values can be adjusted to meet test requirements
    '''
    product = iscetest.data + 'envisat.h5'
    freq = 'A'
    # just like C++ test, use input from rdr2geo workflow test
    topopath = 'rdr2geo/topo.vrt'
    azoff = 0.0
    rgoff = 0.0
    outdir = 'geo2rdr_cpu'

    gpu = False


def testCpuGeo2Rdr():
    '''
    run geo2rdr on CPU
    '''
    # init inputs
    opts = Geo2RdrOpts()
    opts.topopath = 'rdr2geo_cpu/topo.vrt'

    # run resamp
    geo2rdr.main(opts)

    f_offsets = ['azimuth.off', 'range.off']
    # load generated offsets
    for f_offset in f_offsets:
        off_raster = np.fromfile(os.path.join(opts.outdir, f_offset), dtype=np.float32)

        # zero null values
        off_raster = off_raster[np.abs(off_raster) <= 999.0]

        # accumulate error
        off_error = (off_raster * off_raster).sum()

        # check errors
        assert (off_error < 1e-9), f'NISAR Python CPU {f_offset} error out of bounds: {off_error} > 1e-9'


if __name__ == '__main__':
    testCpuGeo2Rdr()

# end of file
