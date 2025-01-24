import iscetest
from nisar.workflows.gen_el_null_range_product import \
    gen_el_null_range_product

import numpy.testing as npt
import argparse
import os


class TestGenElNullRangeProduct:
    # L0B filenames w/ and w/o calib
    # The RX channels are balanced vi RxCal (caltone).
    l0b_filename = 'REE_L0B_CHANNEL4_EXTSCENE_PASS1_LINE3000_CALIB.h5'
    # The RX channels are imbalance!
    l0b_filename_nocal = 'REE_L0B_CHANNEL4_EXTSCENE_PASS1_LINE3000_UNCALIB.h5'
    # Antenna filename
    ant_filename = 'REE_ANTPAT_CUTS_BEAM4.h5'
    # External orbit XML file
    orbit_filename = 'REE_ORBIT_CHANNEL4_EXTSCENE_PASS1.xml'
    # External attitude XML file
    attitude_filename = 'REE_ATTITUDE_CHANNEL4_EXTSCENE_PASS1.xml'

    # set input arguments
    args = argparse.Namespace(
        l0b_file=os.path.join(iscetest.data, l0b_filename),
        antenna_file=os.path.join(iscetest.data, ant_filename),
        freq_band=None, txrx_pol=None, dem_file=None, apply_caltone=False,
        az_block_dur=2.0, out_path='.', ref_height=-100.0,
        orbit_file=None, attitude_file=None, plot=False, polyfit_deg=6,
        exclude_nulls=None
    )

    def test_correct_args(self):
        gen_el_null_range_product(self.args)

    def test_incorrect_args(self):
        # change the frequency band to a non-existing one
        self.args.freq_band = 'B'
        with npt.assert_raises(ValueError):
            gen_el_null_range_product(self.args)

    def test_orbit_attitude_caltone_plot(self):
        self.args.freq_band = 'A'
        self.args.plot = True
        self.args.l0b_file = os.path.join(
            iscetest.data, self.l0b_filename_nocal)
        self.args.apply_caltone = True
        self.args.orbit_file = os.path.join(
            iscetest.data, self.orbit_filename)
        self.args.attitude_file = os.path.join(
            iscetest.data, self.attitude_filename)
        gen_el_null_range_product(self.args)

    def test_exclude_nulls(self):
        # exclude null # 2
        self.args.exclude_nulls = [2]
        gen_el_null_range_product(self.args)
