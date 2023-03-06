import iscetest
from nisar.products.readers.instrument import InstrumentParser

import numpy.testing as npt
import os


def test_instrument_parser():
    # sub directory for test files under "isce3/tests/data"
    sub_dir = 'bf'

    # HDF5 filename for instrument under "sub_dir"
    instrument_file = 'REE_INSTRUMENT_TABLE.h5'

    # construct the parser
    with InstrumentParser(os.path.join(iscetest.data, sub_dir,
                                       instrument_file)) as ins:

        # check some attirbutes per knowledge from the file content
        npt.assert_equal(ins.pols, ['H', 'V'], err_msg='Wrong list of pols')

        npt.assert_equal(ins.num_channels, 12,
                         err_msg='Wrong number of channels')

        npt.assert_equal(ins.num_angles_dbf, 256,
                         err_msg='Wrong number of DBF angles')

        npt.assert_equal(ins.num_sets_dbf, 3,
                         err_msg='Wrong number of DBF sets')

        # parse dataset and check some values or shape of the array
        # for all pols
        for pol in ins.pols:
            # Parse TA table and its sampling rate
            ta = ins.get_time2angle(pol)
            fs_ta = ins.sampling_rate_ta(pol)
            npt.assert_allclose(fs_ta, 96e6,
                                err_msg=f'Wrong TA sampling rate for "{pol}"')
            npt.assert_equal(ta.shape, [12, 256],
                             err_msg=f'Wrong TA shape for "{pol}"')

            # Parse AC table and EL angles
            ac = ins.get_angle2coef(pol)
            el_ang = ins.el_angles_ac(pol)
            npt.assert_equal(ac.shape, [12, 256],
                             err_msg=f'Wrong AC shape for "{pol}"')
            npt.assert_equal(el_ang.shape, [12, 256],
                             err_msg=f'Wrong EL angle shape for "{pol}"')
