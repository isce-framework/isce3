import iscetest
from nisar.workflows.gen_doppler_range_product import gen_doppler_range_product

import numpy.testing as npt
import argparse
import os


class TestGenDopplerRangeProduct:
    # L0B filename
    l0b_file = 'ALPSRP081257070-H1.0__A_HH_2500_LINES.h5'
    ant_file = 'ALOS1_PALSAR_ANTPAT_FIVE_BEAMS.h5'

    # set input arguments
    args = argparse.Namespace(
        filename_l0b=os.path.join(iscetest.data, l0b_file),
        freq_band='A', txrx_pol='HH', num_rgb_avg=32,
        dop_method='CDE', az_block_dur=0.8753, time_interval=0.2918,
        subband=False, polyfit=False, polyfit_deg=3,
        plot=True, out_path='.',
        antenna_file=os.path.join(iscetest.data, ant_file))

    def test_correct_args(self):
        gen_doppler_range_product(self.args)

    def test_incorrect_args(self):
        # change the frequency band to a non-existing one
        self.args.freq_band = 'B'
        with npt.assert_raises(ValueError):
            gen_doppler_range_product(self.args)
