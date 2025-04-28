import iscetest
from nisar.pointing import doppler_lut_from_raw
from nisar.products.readers.Raw import open_rrsd
from isce3.core import speed_of_light

import os
import numpy as np
import numpy.testing as npt


def get_doppler_from_attitude_eb(raw_obj, freq_band, txrx_pol, tm_mid,
                                 eb_angle_deg):
    """
    Estimate doppler at mid pulse time from attitude due to e.g., imperfect
    zero-doppler steering as well as contribution from electrical boresight.

    Parameters
    ----------
    raw_obj : isce3.nisar_products.readers.Raw.RawBase
    freq_band : str
    txrx_pol : str
    tm_mid : float
        Mid azimuth time of echo in (sec)
    eb_angle_deg : float
        Electrtcial boresight angle in (deg)

    Returns
    -------
    float
        doppler centroid in (Hz)
    float
        total squint angle, residual yaw plus EB, in (rad)

    """
    # get orbit and attitude objects
    orb = raw_obj.getOrbit()
    att = raw_obj.getAttitude()

    # get state vectors and quaternions at mid echo time
    pos, vel = orb.interpolate(tm_mid)
    vel_mag = np.linalg.norm(vel)
    quat = att.interpolate(tm_mid)

    # get sign of Y-axis from radar/antenna looking direction
    sgn = {'R': -1}.get(raw_obj.identification.lookDirection[0], 1)

    # get Y-axis with proper sign in ECEF
    y_ecef = quat.rotate([0, sgn, 0])

    # get the yaw angle (rad) due to imperfect zero-doppler steering
    yaw_ang = np.arccos(np.dot(vel / vel_mag, y_ecef))

    # get the wavelength
    wl = speed_of_light / raw_obj.getCenterFrequency(freq_band)

    # total squint angle
    squnit_ang = np.deg2rad(eb_angle_deg) + yaw_ang

    # calculate doppler centroid
    dop_cnt = 2. * vel_mag / wl * np.sin(squnit_ang)

    return dop_cnt, squnit_ang


class TestDopplerLutFromRaw:
    # List of inputs

    # filename of ALOS1 PALSAR data over homogenenous scene like
    # Amazon rainforest
    filename = 'ALPSRP081257070-H1.0__A_HH_2500_LINES.h5'

    # TxRx Polarization of the echo product
    txrx_pol = 'HH'

    # frequency band 'A' or 'B'
    freq_band = 'A'

    # absolute MSE tolerance in Doppler centroid estimation in (Hz)
    dop_cnt_err = 12.0

    # electrical boresight (EB) angle in (deg). EB along with residual
    # Yaw angle defines final squint angle (deviation from zero doppler plane)
    # and thus Doppler centroid.
    eb_angle_deg = 0.0

    # azimuth block duration and time interval in (sec)
    # values are chosen to result in at least 2 azimuth blocks!
    az_block_dur = 0.875
    time_interval = 0.29

    # expected prototype Chebyshev Equi-ripple filter length when subbanding
    # requested in joint time-freq doppler estimation
    filter_length = 33

    # The object/values obtained from the inputs and shared by all methods

    # get ISCE3 Raw object from L0B file
    raw_obj = open_rrsd(os.path.join(iscetest.data, filename))

    # get number of bad values for the first range line
    dset = raw_obj.getRawDataset(freq_band, txrx_pol)[0]
    num_bad_vals = np.sum(np.isnan(dset) | np.isclose(dset, 0))

    # get ref epoch and mid azimuth pulse time of the echo
    ref_epoch, az_tm = raw_obj.getPulseTimes(freq_band, txrx_pol[0])
    tm_mid = az_tm.mean()
    pri = az_tm[1] - az_tm[0]

    # calculate number of azimuth blocks
    _len_tm_int = int(time_interval / pri)
    _len_az_blk_dur = int(az_block_dur / pri)
    num_az_blocks = int(np.ceil((len(az_tm) - _len_az_blk_dur) /
                                _len_tm_int)) + 1

    # get slant range
    sr = raw_obj.getRanges(freq_band, txrx_pol[0])

    # get the expected mean doppler centroid of the echo in (Hz)
    dop_cnt_mean, squint_ang = get_doppler_from_attitude_eb(
        raw_obj, freq_band, txrx_pol, tm_mid, eb_angle_deg)

    def _validate_doppler_lut(self, dop_lut, num_rgb_avg=1, err_msg=''):
        """
        Compare mean, std, ref of Doppler LUT2d values with expected mean
        within "dop_cnt_err". Check the shape and axes of the LUT2d.

        Parameters
        ----------
        dop_lut : isce3.core.LUT2d
            Estimated Doppler LUT
        num_rgb_avg : int, default=1
            Number of range bins to be averaged.
        err_msg : str, default=''

        """
        # check the shape of LUT
        num_sr = self.sr.size // num_rgb_avg
        lut_shape = (self.num_az_blocks, num_sr)
        npt.assert_equal(
            dop_lut.data.shape,  lut_shape,
            err_msg=f'Wrong shape of LUT2d {err_msg}'
        )
        # check statistics of the LUT
        npt.assert_allclose(
            abs(dop_lut.data.mean() - self.dop_cnt_mean), 0.0,
            atol=self.dop_cnt_err,
            err_msg='Mean Doppler centroids from Raw exceeds error '
            f'{self.dop_cnt_err} (Hz) {err_msg}'
        )

        npt.assert_allclose(
            dop_lut.data.std(), 0.0, atol=self.dop_cnt_err,
            err_msg='STD of Doppler centroids from Raw exceeds error '
            f'{self.dop_cnt_err} (Hz) {err_msg}'
        )
        # check the start and spacing for x-axis (slant range) in (m)
        spacing_sr = num_rgb_avg * self.sr.spacing
        npt.assert_allclose(
            dop_lut.x_spacing, spacing_sr,
            err_msg=f'Wrong slant range/X spacing of LUT {err_msg}'
        )

        start_sr = self.sr[num_rgb_avg // 2]
        npt.assert_allclose(
            dop_lut.x_start, start_sr,
            err_msg=f'Wrong start slant range/X of LUT {err_msg}'
        )
        # check azimuth time/y-axis start and spcaing by using PRI
        # as absolute tol
        start_az = self.az_tm[0] + self.az_block_dur / 2.0
        npt.assert_allclose(
            dop_lut.y_spacing, self.time_interval, atol=self.pri,
            err_msg=f'Wrong spcaing az time/Y of LUT {err_msg}'
        )
        npt.assert_allclose(
            dop_lut.y_start, start_az, atol=self.pri,
            err_msg=f'Wrong start az time/Y of LUT {err_msg}'
        )

    def test_doppler_est_time(self):
        # print expected doppler centroid and squint angle values
        print(
            'Expected mean squint angle from attitude plus EB  -> '
            f'{np.rad2deg(self.squint_ang) * 1e3:.1f} (mdeg)'
        )
        print(
            'Expected mean Doppler centroid from attitude plus EB -> '
            f'{self.dop_cnt_mean:.2f} (Hz)'
        )
        num_rgb_avg = 4
        # estimate Doppler LUT
        dop_lut, dop_epoch, dop_mask, corr_coef, dop_pol, centerfreq, \
            dop_flt_coef = doppler_lut_from_raw(
                self.raw_obj, num_rgb_avg=num_rgb_avg,
                az_block_dur=self.az_block_dur,
                time_interval=self.time_interval)

        # validate center freq
        npt.assert_allclose(centerfreq, self.raw_obj.getCenterFrequency(
            self.freq_band, self.txrx_pol[0]),
            err_msg='Wrong center frequency')
        # validate pol
        npt.assert_equal(dop_pol, self.txrx_pol, err_msg='Wrong TxRx Pol')
        # validate epoch
        npt.assert_equal(dop_epoch, self.ref_epoch, err_msg='Wrong Ref epoch')
        # validate Doppler LUT axes, shape, statistics
        self._validate_doppler_lut(dop_lut, num_rgb_avg=num_rgb_avg,
                                   err_msg=' in time approach')
        # validate mask array shape and values
        npt.assert_equal(dop_mask.shape, dop_lut.data.shape,
                         err_msg='Wrong shape of mask array')
        # get total number of bad items for all azimuth blocks
        tot_bad_vals = (self.num_bad_vals // num_rgb_avg) * dop_mask.shape[0]
        # check the mask array False values against the expected one
        npt.assert_equal(np.where(~dop_mask)[0].size, tot_bad_vals,
                         err_msg='Wrong number of False values in mask array')
        # validate correlation coeffs shape and values
        npt.assert_equal(corr_coef.shape, dop_lut.data.shape,
                         err_msg='Wrong shape of correlation coeffs')

        npt.assert_equal(np.all(corr_coef >= 0) and np.all(corr_coef <= 1),
                         True, err_msg='Correlation coeffs are out of range'
                         ' [0,1]')
        # check filter coeffs if any
        npt.assert_equal(dop_flt_coef, None,
                         err_msg='Existence of filter coeffs in time approach')

    def test_doppler_est_time_subband(self):
        # estimate Doppler LUT
        dop_lut, _, _, _, _, _, dop_flt_coef = doppler_lut_from_raw(
            self.raw_obj, az_block_dur=self.az_block_dur,
            time_interval=self.time_interval, subband=True)
        # validate Doppler LUT axes, shape, statistics
        self._validate_doppler_lut(dop_lut, num_rgb_avg=8,
                                   err_msg=' in joint time-frequency approach')
        # check filter coeffs
        npt.assert_equal(
            dop_flt_coef.size, self.filter_length,
            err_msg='Wrong filter length in time-frequency approach'
        )

    def test_doppler_est_time_polyfit(self):
        # estimate Doppler LUTf
        dop_lut, _, _, _, _, _, _ = doppler_lut_from_raw(
            self.raw_obj, az_block_dur=self.az_block_dur,
            time_interval=self.time_interval, polyfit=True)
        # validate Doppler LUT axes, shape, statistics
        self._validate_doppler_lut(
            dop_lut, num_rgb_avg=8,
            err_msg=' in time approach with polyfitted output'
        )
