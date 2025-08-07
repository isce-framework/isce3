import os
import argparse
import copy

from nisar.workflows.faraday_rot_angle_from_rslc import \
    faraday_rot_angle_from_rslc
import iscetest


class TestFaradayRotAngleFromRSlc:
    # List of inputs
    sub_dir = 'faraday_rot'

    # Linear quad-pol RSLC product over homogeneous extended scene
    # and corner reflector(s)
    file_slc = 'pol_rad_calib_rslc_Remningstorp_ALPSRP030392430.h5'

    # UAVSAR-formatted CSV file for corner reflector(s) in the above
    # RSLC product
    file_csv = 'Corner_Reflector_Remningstorp_ALPSRP030392430.csv'
    # NISAR-formatted CSV file for corner reflector(s) in the above
    # RSLC product
    file_csv_nisar = 'Corner_Reflector_Remningstorp_ALPSRP030392430_NISAR.csv'

    # form the file-path names
    f_slc = os.path.join(iscetest.data, sub_dir, file_slc)
    f_csv = os.path.join(iscetest.data, sub_dir, file_csv)
    f_csv_nisar = os.path.join(iscetest.data, sub_dir, file_csv_nisar)

    # build common/default input args
    args = argparse.Namespace(
        slc=f_slc, csv_cr=None, freq_band='A', method='BB',
        out_dir='.', average=False, sr_lim=(None, None),
        azt_lim=(None, None), sr_spacing=3000, azt_spacing=5.0,
        plot=False
    )

    def test_default_args(self):
        faraday_rot_angle_from_rslc(self.args)

    def test_average_freeman_second(self):
        args = copy.copy(self.args)
        args.average = True
        args.method = 'FS'
        faraday_rot_angle_from_rslc(args)

    def test_freq_slope_plot(self):
        args = copy.copy(self.args)
        args.method = 'SLOPE'
        args.plot = True
        faraday_rot_angle_from_rslc(args)

    def test_with_corner_reflector_uavsar(self):
        args = copy.copy(self.args)
        args.csv_cr = self.f_csv
        faraday_rot_angle_from_rslc(args)

    def test_with_corner_reflector_nisar(self):
        args = copy.copy(self.args)
        args.csv_cr = self.f_csv_nisar
        faraday_rot_angle_from_rslc(args)
