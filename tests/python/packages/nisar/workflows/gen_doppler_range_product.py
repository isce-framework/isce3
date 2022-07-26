import iscetest
from nisar.workflows.gen_doppler_range_product import gen_doppler_range_product

import numpy.testing as npt
import argparse
import os


class TestGenDopplerRangeProduct:
    # L0B and antenna file (beam # 7) for ALOS1 data over Amazon
    # used for argument testing with single-channel antenna w/ plotting given
    # it is homogenous scene and suitable for a good doppler estimation.
    l0b_alos1 = 'ALPSRP081257070-H1.0__A_HH_2500_LINES.h5'
    ant_alos1 = 'pointing/ALOS1_PALSAR_ANTPAT_BEAM343.h5'

    # L0B, orbit, attitude, antenna files from NISAR-like science mode (DBFed)
    # used for testing with external orbit, attitude and multi-channel antenna
    # plus polyfitting option.
    # Note that this dataset is over heterogenous scene. The antenna is also
    # steered to 0.3 deg rather than 0.9 deg; reported in the antenna file;
    # due to resource limitation at the time of simulation. That being said,
    # the reported doppler from attitude+antenna (~900 Hz) shall be around
    # three times the one estimated from the raw echo (~300 Hz)!
    sub_dir = 'pointing'
    l0b_nisar = 'REE_L0B_DBF_EXTSCENE_PASS1_LINE3000_TRUNCATED.h5'
    ant_nisar = 'REE_ANTPAT_CUTS_DBF.h5'
    orb_nisar = 'REE_ORBIT_DATA_DBF_PASS1.xml'
    att_nisar = 'REE_ATTITUDE_DATA_DBF_PASS1.xml'

    # set input arguments for ALOS1 case
    args_alos1 = argparse.Namespace(
        filename_l0b=os.path.join(iscetest.data, l0b_alos1),
        freq_band='A', txrx_pol='HH', num_rgb_avg=32,
        dop_method='CDE', az_block_dur=0.8753, time_interval=0.2918,
        subband=False, polyfit=False, polyfit_deg=3,
        plot=True, out_path='.', orbit_file=None, attitude_file=None,
        antenna_file=os.path.join(iscetest.data, ant_alos1))

    # set input arguments for NISAR case
    args_nisar = argparse.Namespace(
        filename_l0b=os.path.join(iscetest.data, sub_dir, l0b_nisar),
        freq_band='A', txrx_pol='VV', num_rgb_avg=16,
        dop_method='CDE', az_block_dur=1.0, time_interval=0.5,
        subband=False, polyfit=True, polyfit_deg=3, plot=False, out_path='.',
        orbit_file=os.path.join(iscetest.data, sub_dir, orb_nisar),
        attitude_file=os.path.join(iscetest.data, sub_dir, att_nisar),
        antenna_file=os.path.join(iscetest.data, sub_dir, ant_nisar))

    def test_correct_args(self):
        # use ALOS1 set for this test
        gen_doppler_range_product(self.args_alos1)

    def test_incorrect_args(self):
        # use ALOS1 set for this test
        # change the frequency band to a non-existing one
        self.args_alos1.freq_band = 'B'
        with npt.assert_raises(ValueError):
            gen_doppler_range_product(self.args_alos1)

    def test_ext_orbit_att_polyfit(self):
        # use NISAR set for this test
        gen_doppler_range_product(self.args_nisar)
