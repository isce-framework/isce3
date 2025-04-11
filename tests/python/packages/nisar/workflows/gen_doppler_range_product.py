import iscetest
from nisar.workflows.gen_doppler_range_product import gen_doppler_range_product

import numpy.testing as npt
import argparse
import os
import warnings


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

    # DM2 L0B product with dem raster file.
    # This is truncated 6-channel NISAR-like DM2 over Amazon scene
    # w/ topography
    subdir_dm2 = 'dm2'
    l0b_dm2 = 'REE_L0B_AMAZON_PASS1_RGL_4500-9500_RGB_10-2440.h5'
    ant_dm2 = 'REE_ANTPAT_CUTS_AMAZON.h5'
    dem_dm2 = 'REE_DEM_AMAZON_PASS1.tif'

    # set input arguments for ALOS1 case
    args_alos1 = argparse.Namespace(
        filename_l0b=os.path.join(iscetest.data, l0b_alos1),
        freq_band='A', txrx_pol='HH', num_rgb_avg=32, ref_height=0.0,
        dop_method='CDE', az_block_dur=0.8753, time_interval=0.2918,
        subband=False, polyfit=False, polyfit_deg=3, dem_file=None,
        plot=True, out_path='.', orbit_file=None,
        attitude_file=None, exclude_beams=None,
        antenna_file=os.path.join(iscetest.data, ant_alos1))

    # set input arguments for DBF NISAR case
    args_nisar = argparse.Namespace(
        filename_l0b=os.path.join(iscetest.data, sub_dir, l0b_nisar),
        freq_band=None, txrx_pol='VV', num_rgb_avg=16, ref_height=0.0,
        dop_method='CDE', az_block_dur=1.0, time_interval=0.5, dem_file=None,
        subband=False, polyfit=True, polyfit_deg=3, plot=False,
        out_path='.', exclude_beams=None,
        orbit_file=os.path.join(iscetest.data, sub_dir, orb_nisar),
        attitude_file=os.path.join(iscetest.data, sub_dir, att_nisar),
        antenna_file=os.path.join(iscetest.data, sub_dir, ant_nisar))

    # set input arguments for DM2 NISAR case
    args_dm2 = argparse.Namespace(
        filename_l0b=os.path.join(iscetest.data, subdir_dm2, l0b_dm2),
        freq_band=None, txrx_pol=None, num_rgb_avg=8, ref_height=0.0,
        dop_method='CDE', az_block_dur=2.5, time_interval=0.11,
        subband=False, polyfit=True, polyfit_deg=3, exclude_beams=[3],
        plot=False, out_path='.', orbit_file=None, attitude_file=None,
        antenna_file=os.path.join(iscetest.data, subdir_dm2, ant_dm2),
        dem_file=os.path.join(iscetest.data, subdir_dm2, dem_dm2))

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
        # use NISAR DBF set for this test
        gen_doppler_range_product(self.args_nisar)

    def test_dm2_dem_ant(self):
        # use NISAR DM2 case for this test
        # check if subdir "dm2" exists and then
        # run the test
        dm2_dir = os.path.join(iscetest.data, self.subdir_dm2)
        if os.path.exists(dm2_dir):
            gen_doppler_range_product(self.args_dm2)
        else:
            warnings.warn(
                f'Subdir "{self.subdir_dm2}" with DM2 files does not exist!')
