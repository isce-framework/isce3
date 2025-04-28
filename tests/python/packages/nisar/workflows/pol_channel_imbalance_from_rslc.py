import os
import argparse
import numpy.testing as npt
import numpy as np
import copy

from nisar.workflows.pol_channel_imbalance_from_rslc import \
    pol_channel_imbalance_from_rslc
from nisar.cal import OutOfSlcBoundError
import iscetest


class TestPolChannelImbalanceFromRSlc:
    # List of inputs
    sub_dir = 'pol_cal'

    # RSLC product over homogeneous extended scene
    file_slc_ext = 'calib_RSLC_ALPSRP277307130_AMAZON.h5'
    # RSLC product over corner reflector(s)
    file_slc_cr = 'calib_RSLC_ALPSRP025826990_RIO_BRANCO_CR.h5'
    # UAVSAR-formatted CSV file for Corner reflector(s)
    file_csv_cr = 'Corner_Reflector_Rio_Branco_ALPSRP025826990.csv'
    # NISAR-formatted CSV file for Corner reflector(s)
    file_csv_cr_nisar = 'Corner_Reflector_Rio_Branco_ALPSRP025826990_NISAR.csv'

    # cross-talk magnitudes at three EL angles
    # Tx [H, V] cross talk ratios (~ -35 dB)
    # Rx [H, V] cross talk ratios (~ -35 dB)
    el_xtalk_deg = [-1, 0, 1]
    tx_xpol_h = [0.017, 0.016, 0.018]
    tx_xpol_v = [0.017, 0.016, 0.018]
    rx_xpol_h = [0.017, 0.016, 0.018]
    rx_xpol_v = [0.017, 0.016, 0.018]

    # form the full filenames
    f_slc_ext = os.path.join(iscetest.data, sub_dir, file_slc_ext)
    f_slc_cr = os.path.join(iscetest.data, sub_dir, file_slc_cr)
    f_csv_cr = os.path.join(iscetest.data, sub_dir, file_csv_cr)
    f_csv_cr_nisar = os.path.join(iscetest.data, sub_dir, file_csv_cr_nisar)

    # form common/default input args
    args = argparse.Namespace(
        slc_ext=f_slc_ext, slc_cr=f_slc_cr, csv_cr=f_csv_cr, freq_band='A',
        dem_file=None, ref_height=0.0, out_dir='.', ignore_faraday=False,
        tx_xtalk_amp_h=[0.0], tx_xtalk_phs_h=None, tx_xtalk_amp_v=[0.0],
        tx_xtalk_phs_v=None, rx_xtalk_amp_h=[0.0], rx_xtalk_phs_h=None,
        rx_xtalk_amp_v=[0.0], rx_xtalk_phs_v=None, el_xtalk=[0],
        sr_lim=(None, None), azt_lim=(None, None), sr_spacing=3000,
        azt_spacing=5.0, mean_el=False, plot=False
    )

    def test_default_args_uavsar_csv(self):
        pol_channel_imbalance_from_rslc(self.args)

    def test_xtalk_no_faraday(self):
        args = copy.copy(self.args)
        # Set the magnitude of X-talks.
        # Phases will be automatically set to zeros.
        args.tx_xtalk_amp_h = self.tx_xpol_h
        args.tx_xtalk_amp_v = self.tx_xpol_v
        args.rx_xtalk_amp_h = self.rx_xpol_h
        args.rx_xtalk_amp_v = self.rx_xpol_v
        args.el_xtalk = np.deg2rad(self.el_xtalk_deg)
        args.ignore_faraday = True
        pol_channel_imbalance_from_rslc(args)

    def test_out_of_slc_bound_error(self):
        args = copy.copy(self.args)
        args.azt_lim = (12217.0, 12227.0)
        with npt.assert_raises(OutOfSlcBoundError):
            pol_channel_imbalance_from_rslc(args)

    def test_mean_el_and_plot(self):
        args = copy.copy(self.args)
        args.mean_el = True
        args.plot = True
        pol_channel_imbalance_from_rslc(args)

    def test_default_args_nisar_csv(self):
        args = copy.copy(self.args)
        args.csv_cr = self.f_csv_cr_nisar
        pol_channel_imbalance_from_rslc(self.args)
