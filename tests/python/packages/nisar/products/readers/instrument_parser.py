import iscetest
from nisar.products.readers.instrument import (
    InstrumentParser, MissingInstrumentFieldWarning
)

import numpy.testing as npt
import numpy as np
import os


def amp2db(amp: complex) -> float:
    """Complex amplitude to dB"""
    return 20 * np.log10(abs(amp))


def test_instrument_parser_v1p0():
    # sub directory for test files under "isce3/tests/data"
    sub_dir = 'bf'

    # HDF5 filename for instrument under "sub_dir"
    instrument_file = 'REE_INSTRUMENT_TABLE.h5'

    # default values for channel adjustment factors if missing
    adj_fact_default = None

    # default cross-talk value at any EL angle if missing
    xtalk_default = complex()

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

            # parse channel adjustment factors for TX and RX
            # All values are 1.0 for missing field "channelAdjustment" in v1.0
            # Expected custom warning "MissingInstrumentFieldWarning"!
            with npt.assert_warns(MissingInstrumentFieldWarning):
                tx_adj_fact = ins.channel_adjustment_factors_tx(pol)
                npt.assert_equal(tx_adj_fact, adj_fact_default,
                                 err_msg=('Wrong TX channel adjustment'
                                          f' factors for "{pol}"')
                                 )

                rx_adj_fact = ins.channel_adjustment_factors_rx(pol)
                npt.assert_equal(rx_adj_fact, adj_fact_default,
                                 err_msg=('Wrong RX channel adjustment'
                                          f' factors for "{pol}"')
                                 )

        # parse cross-talk
        # Expected custom warning "MissingInstrumentFieldWarning"!
        with npt.assert_warns(MissingInstrumentFieldWarning):
            xtalk = ins.get_crosstalk()
            # all x-talks are zero for missing field "crossTalk" in v1.0
            npt.assert_allclose(xtalk.tx_xpol_h(0), xtalk_default,
                                err_msg='Wrong H-pol TX X-talk value at EL=0!')
            npt.assert_allclose(xtalk.tx_xpol_v(0), xtalk_default,
                                err_msg='Wrong V-pol TX X-talk value at EL=0!')
            npt.assert_allclose(xtalk.rx_xpol_h(0), xtalk_default,
                                err_msg='Wrong H-pol RX X-talk value at EL=0!')
            npt.assert_allclose(xtalk.rx_xpol_v(0), xtalk_default,
                                err_msg='Wrong V-pol RX X-talk value at EL=0!')


def test_instrument_parser_v2p0():
    """
    Parsing V2.0 of NISAR instrument table where two new fields
    "channelAdjustment" and "crossTalk" are added to its older version V1.0.

    """
    # sub directory for test files under "isce3/tests/data"
    sub_dir = 'bf'

    # HDF5 filename for instrument under "sub_dir"
    instrument_file = 'REE_INSTRUMENT_TABLE_V2P0.h5'

    # max expected EL X-talk ratio extracted from its antenna patterns
    xtalk_max_db = -39.0

    # number of EL points in the cross talk object
    num_el = 581

    # channel adjustment factors stored in the file
    adj_factors = np.ones(12, dtype='c16')

    # construct the parser
    with InstrumentParser(os.path.join(iscetest.data, sub_dir,
                                       instrument_file)) as ins:

        # check out the cross talk
        xtalk = ins.get_crosstalk()

        npt.assert_equal(xtalk.tx_xpol_h.x.size, num_el,
                         err_msg='Wrong size for H-pol TX X-talk')
        npt.assert_(amp2db(xtalk.tx_xpol_h(0)) <= xtalk_max_db,
                    msg='Wrong H-pol TX X-talk value at EL=0!')

        npt.assert_equal(xtalk.tx_xpol_v.x.size, num_el,
                         err_msg='Wrong size for V-pol TX X-talk')
        npt.assert_(amp2db(xtalk.tx_xpol_v(0)) <= xtalk_max_db,
                    msg='Wrong V-pol TX X-talk value at EL=0!')

        npt.assert_equal(xtalk.rx_xpol_h.x.size, num_el,
                         err_msg='Wrong size for H-pol RX X-talk')
        npt.assert_(amp2db(xtalk.rx_xpol_h(0)) <= xtalk_max_db,
                    msg='Wrong H-pol RX X-talk value at EL=0!')

        npt.assert_equal(xtalk.rx_xpol_v.x.size, num_el,
                         err_msg='Wrong size for V-pol RX X-talk')
        npt.assert_(amp2db(xtalk.rx_xpol_v(0)) <= xtalk_max_db,
                    msg='Wrong V-pol RX X-talk value at EL=0!')

        # check the adjustment factors which are all set to 1.0
        for pol in ('H', 'V'):
            tx_adj_fact = ins.channel_adjustment_factors_tx(pol)
            npt.assert_allclose(tx_adj_fact, adj_factors,
                                err_msg='Wrong TX channel adjustment factors'
                                f' for "{pol}"')

            rx_adj_fact = ins.channel_adjustment_factors_rx(pol)
            npt.assert_allclose(rx_adj_fact, adj_factors,
                                err_msg='Wrong RX channel adjustment factors'
                                f' for "{pol}"')
