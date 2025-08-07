import iscetest
from nisar.workflows.gen_el_rising_edge_product import \
    gen_el_rising_edge_product

import numpy.testing as npt
import argparse
import os


class TestGenElRisingEdgeProductNisar:
    # subdirectory for all files
    subdir = 'pointing'
    # L0B filenames
    l0b_filename = 'REE_L0B_DBF_EXTSCENE_PASS1_LINE3000_TRUNCATED.h5'
    # Antenna filename
    ant_filename = 'REE_ANTPAT_CUTS_DBF.h5'
    # External orbit XML file
    orbit_filename = 'REE_ORBIT_DATA_DBF_PASS1.xml'
    # External attitude XML file
    attitude_filename = 'REE_ATTITUDE_DATA_DBF_PASS1.xml'

    # set input arguments
    args = argparse.Namespace(
        l0b_file=os.path.join(iscetest.data, subdir, l0b_filename),
        antenna_file=os.path.join(iscetest.data, subdir, ant_filename),
        freq_band=None, txrx_pol=None, dem_file=None, no_dbf_norm=False,
        az_block_dur=2.0, out_path='.', ref_height=0.0, plot=False,
        orbit_file=None, attitude_file=None, no_weight=False,
    )

    def test_correct_args(self):
        gen_el_rising_edge_product(self.args)

    def test_incorrect_args(self):
        # change the frequency band to a non-existing one
        self.args.freq_band = 'B'
        with npt.assert_raises(ValueError):
            gen_el_rising_edge_product(self.args)

    def test_no_weight(self):
        self.args.freq_band = 'A'
        self.args.no_weight = True
        gen_el_rising_edge_product(self.args)

    def test_ext_orbit_attitude(self):
        self.args.orbit_file = os.path.join(
            iscetest.data, self.subdir, self.orbit_filename)
        self.args.attitude_file = os.path.join(
            iscetest.data, self.subdir, self.attitude_filename)
        self.args.plot = True
        self.args.no_weight = False
        gen_el_rising_edge_product(self.args)


def test_gen_el_rising_edge_product_alos():
    # subdirectory for all files
    subdir = 'pointing'
    # ALOS1 L0B filename over amazon, beam # 7
    l0b_filename = 'ALPSRP264757150-H1.0__A_HH_LINE4000-5000_RANGE0-2500.h5'
    # Antenna filename
    ant_filename = 'ALOS1_PALSAR_ANTPAT_BEAM343.h5'
    # Ref/mean height in (m) over most part of Amazon
    ref_hgt = 100.0
    # TxRx Pol of the product
    txrx_pol = 'HH'
    # set input arguments
    args = argparse.Namespace(
        l0b_file=os.path.join(iscetest.data, subdir, l0b_filename),
        antenna_file=os.path.join(iscetest.data, subdir, ant_filename),
        freq_band='A', txrx_pol=txrx_pol, dem_file=None, no_dbf_norm=False,
        az_block_dur=5.0, out_path='.', ref_height=ref_hgt, plot=True,
        orbit_file=None, attitude_file=None, no_weight=False, beam_num=1,
    )
    # run product generator
    gen_el_rising_edge_product(args)
