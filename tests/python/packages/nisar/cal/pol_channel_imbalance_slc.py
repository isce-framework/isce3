import os
import numpy as np
import numpy.testing as npt
from scipy.interpolate import interp1d

import iscetest
from nisar.cal import PolChannelImbalanceSlc
from nisar.products.readers.SLC import SLC
from isce3.antenna import CrossTalk
from nisar.workflows.pol_channel_imbalance_from_rslc import cr_llh_from_csv


class TestPolChannelImbalanceSlc:
    """

    References
    ----------
    .. [1] M. Shimada, M. Ohki, "Calibration of PALSAR Polarimetry Using
        Corner Reflectors In the Amazon Forest and Its Three-Year Trend,"
        Workshop on Science and Applications of SAR Polarimetry and PolInSAR,
        Italy, April 2009.

    """
    # List of inputs
    sub_dir = 'pol_cal'

    # true values and abs error tols are based on figure(2) in [1]
    # f1 and f2 in [1] are related to tx_imb and rx_imb, respectively.
    # mean amplitudes of Tx, RX imbalances in (linear)
    tx_imb_amp = 1.015
    rx_imb_amp = 0.725
    # mean phases of Tx, RX imbalances in (deg)
    tx_imb_phs = 20.287
    rx_imb_phs = -3.174
    # peak2peak max abs amplitude error tolerance (linear) for both TX and RX
    amp_p2p_atol = 0.13
    # peak2peak max abs phase error tolerance (deg) for both TX and RX
    phs_p2p_atol = 5.0

    # tolerance for amplitude and phase of Rx/Tx ratio (f2/f1 ratio)
    rx2tx_amp = rx_imb_amp / tx_imb_amp
    rx2tx_phs = rx_imb_phs - tx_imb_phs
    amp_ratio_atol = amp_p2p_atol
    phs_ratio_atol = phs_p2p_atol

    # RSLC product over homogenous extended scene
    file_slc_ext = 'calib_RSLC_ALPSRP277307130_AMAZON.h5'
    # RSLC product over corner reflector(s)
    file_slc_cr = 'calib_RSLC_ALPSRP025826990_RIO_BRANCO_CR.h5'
    # UAVSAR-formatted CSV file for Corner reflector(s)
    file_csv_cr = 'Corner_Reflector_Rio_Branco_ALPSRP025826990.csv'
    # NISAR-formatted CSV file for Corner reflector(s)
    file_csv_cr_nisar = 'Corner_Reflector_Rio_Branco_ALPSRP025826990_NISAR.csv'

    # Use Averaged cross-talk magnitudes based on figure (4) in [1]
    # X-talks are simply provided at three Antenna EL angles.
    # Tx [H, V] cross talk ratios (~ -35 dB) as a function EL angles
    # Rx [H, V] cross talk ratios (~ -35 dB) as a function EL angles
    # The half of x-talks has phase around 0 deg while the other half
    # has phase around 180 deg per figure (2) in [1]. To test both
    # real and complex type inputs, both formats are exercised.
    el_xtalk_deg = [-1, 0, 1]
    tx_xpol_h = [
        complex(0.018, 0.0),
        complex(0.018, 0.0),
        complex(0.018, 0.0)
        ]
    tx_xpol_v = [
        -0.018,
        -0.018,
        -0.018
        ]
    rx_xpol_h = [
        0.018,
        0.018,
        0.018
        ]
    rx_xpol_v = [
        complex(-0.018, 0.0),
        complex(-0.018, 0.0),
        complex(-0.018, 0.0)
        ]
    # Form Cross talk object
    el_rad = np.deg2rad(el_xtalk_deg)
    kw_args = dict(kind='linear', fill_value='extrapolate', assume_sorted=True)
    xtalk = CrossTalk(
        interp1d(el_rad, tx_xpol_h, **kw_args),
        interp1d(el_rad, tx_xpol_v, **kw_args),
        interp1d(el_rad, rx_xpol_h, **kw_args),
        interp1d(el_rad, rx_xpol_v, **kw_args)
    )

    # Parse RSLCs
    slc_ext = SLC(hdf5file=os.path.join(iscetest.data, sub_dir, file_slc_ext))

    if file_slc_cr != file_slc_ext:
        slc_cr = SLC(hdf5file=os.path.join(iscetest.data, sub_dir,
                                           file_slc_cr))
    else:
        slc_cr = slc_ext

    # Parse the same CRs from two different formats UAVSAR and NISAR
    cr_llh = cr_llh_from_csv(os.path.join(iscetest.data, sub_dir, file_csv_cr))
    cr_llh_nisar = cr_llh_from_csv(
        os.path.join(iscetest.data, sub_dir, file_csv_cr_nisar)
        )

    def _validate(self, imb_prod_lut1d, imb_prod_slc):
        # Tx Mean Amp/Phase for 2-D SLC-grid product
        tx_mean = imb_prod_slc.tx_pol_ratio.mean()
        npt.assert_allclose(
            abs(tx_mean), self.tx_imb_amp, atol=self.amp_p2p_atol,
            err_msg='Wrong TX Mean Imbalance Amp!')
        npt.assert_allclose(
            np.angle(tx_mean, deg=True), self.tx_imb_phs,
            atol=self.phs_p2p_atol,
            err_msg='Wrong TX Mean Imbalance Phase!')

        # Tx interpolated value at boresight (EL=0.0) using LUT1d product
        tx_br = imb_prod_lut1d.tx_pol_ratio(0)
        npt.assert_allclose(
            abs(tx_br), self.tx_imb_amp, atol=self.amp_p2p_atol,
            err_msg='Wrong TX Imbalance Amp at boresight!')
        npt.assert_allclose(
            np.angle(tx_br, deg=True), self.tx_imb_phs, atol=self.phs_p2p_atol,
            err_msg='Wrong TX Imbalance Phase at boresight!')

        # Rx Mean Amp/Phase for 2-D SLC-grid product
        rx_mean = imb_prod_slc.rx_pol_ratio.mean()
        npt.assert_allclose(
            abs(rx_mean), self.rx_imb_amp, atol=self.amp_p2p_atol,
            err_msg='Wrong RX Mean Imbalance Amp!')
        npt.assert_allclose(
            np.angle(rx_mean, deg=True), self.rx_imb_phs,
            atol=self.phs_p2p_atol,
            err_msg='Wrong RX Mean Imbalance Phase!')

        # Rx interpolated value at boresight (EL=0.0) using LUT1d product
        rx_br = imb_prod_lut1d.rx_pol_ratio(0)
        npt.assert_allclose(
            abs(rx_br), self.rx_imb_amp, atol=self.amp_p2p_atol,
            err_msg='Wrong RX Imbalance Amp at boresight!')
        npt.assert_allclose(
            np.angle(rx_br, deg=True), self.rx_imb_phs, atol=self.phs_p2p_atol,
            err_msg='Wrong RX Imbalance Phase at boresight!')

        # check Rx/Tx ratio from mean value
        rx2tx_mean = rx_mean / tx_mean
        npt.assert_allclose(
            abs(rx2tx_mean), self.rx2tx_amp, atol=self.amp_ratio_atol,
            err_msg='Wrong Amp of mean imbalance ratio of RX/TX!')
        npt.assert_allclose(
            np.angle(rx2tx_mean, deg=True), self.rx2tx_phs,
            atol=self.phs_ratio_atol,
            err_msg='Wrong Phase of mean imbalance ratio of RX/TX!')

        # check Rx/Tx ratio at the boresight
        rx2tx_br = rx_br / tx_br
        npt.assert_allclose(
            abs(rx2tx_br), self.rx2tx_amp, atol=self.amp_ratio_atol,
            err_msg='Wrong Amp of imbalance ratio of RX/TX at boresight!')
        npt.assert_allclose(
            np.angle(rx2tx_br, deg=True), self.rx2tx_phs,
            atol=self.phs_ratio_atol,
            err_msg='Wrong Phase of imbalance ratio of RX/TX at boresight!')

    def test_cr_llh_uavsar_vs_nisar(self):
        npt.assert_allclose(self.cr_llh, self.cr_llh_nisar,
                            err_msg='Different LLH of a CR from UAVSAR and'
                            ' NISAR formatted files!')

    def test_no_xtalk_no_faraday(self):
        with PolChannelImbalanceSlc(
                self.slc_ext, self.slc_cr, self.cr_llh,
                ignore_faraday=True) as pci:
            imb_prod_lut1d, imb_prod_slc = pci.estimate()
        self._validate(imb_prod_lut1d, imb_prod_slc)

    def test_xtalk_no_faraday(self):
        with PolChannelImbalanceSlc(
                self.slc_ext, self.slc_cr, self.cr_llh, cross_talk=self.xtalk,
                ignore_faraday=True) as pci:
            imb_prod_lut1d, imb_prod_slc = pci.estimate()
        self._validate(imb_prod_lut1d, imb_prod_slc)

    def test_xtalk_faraday(self):
        with PolChannelImbalanceSlc(
            self.slc_ext, self.slc_cr, self.cr_llh, cross_talk=self.xtalk,
        ) as pci:
            imb_prod_lut1d, imb_prod_slc = pci.estimate()
        self._validate(imb_prod_lut1d, imb_prod_slc)
