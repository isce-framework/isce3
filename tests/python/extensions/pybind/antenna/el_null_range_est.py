#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt
import os

from isce3.ext.isce3.antenna import ElNullRangeEst, ant2rgdop
from nisar.products.readers.antenna.antenna_parser import AntennaParser
from nisar.products.readers.Raw import Raw
from isce3.ext.isce3.geometry import DEMInterpolator
from isce3.ext.isce3.core import TimeDelta
from isce3.core import speed_of_light
import iscetest


# Test Fixture
class TestElNullRangeEst:
    # Tolerances for EL angle and Doppler at null location
    # from expected (antenna) versus that of measured (echo)
    atol_el_err_mdeg = 10.0  # (mdeg)
    atol_dop = 1  # (Hz)

    # Multi-channel Raw Echo (L0B) and respective antenna files
    # The L0B file is truncated first pass of REE repeated-pass
    # interferometric NISAR-LIKE 4-channel non-DBF (DM2-like)
    # simulation over a very heterogenous scene with good SNR to
    # prove that this Null formation technique almost independent
    # of target reflectivity variations provided decent SNR.
    # The antenna file contains respective four NISAR nominal EL cut
    # patterns used in that REE L0B sim.
    _l0b_file = "REE_L0B_CHANNEL4_EXTSCENE_PASS1_LINE3000_CALIB.h5"
    _ant_file = "REE_ANTPAT_CUTS_BEAM4.h5"

    # DEM mean ref used in sim echo L0B
    _dem_ref = 0.0  # (m)

    # build DEM object
    dem_interp = DEMInterpolator(_dem_ref)

    # parse L0B to get orbit, attitude, echo data
    _raw_obj = Raw(hdf5file=os.path.join(iscetest.data, _l0b_file))

    # get the polarization of TX and RX for freq band 'A'
    txrx_pol = _raw_obj.polarizations.get('A')[0]

    # get chirp parameters and center freq
    center_freq, _, chirp_rate, chirp_dur = _raw_obj.getChirpParameters(
        'A', txrx_pol[0])
    wavelength = speed_of_light / center_freq

    # get slant range spacing
    sr_linspace = _raw_obj.getRanges('A', txrx_pol[0])
    sr_spacing = sr_linspace.spacing
    sr_start = sr_linspace.first

    # get orbit and attitude object
    orbit = _raw_obj.getOrbit()
    attitude = _raw_obj.getAttitude()

    # parse antenna EL cut
    _ant_obj = AntennaParser(os.path.join(iscetest.data, _ant_file))

    # get number of RX beams matched the Pol of Echo
    num_beam = _ant_obj.num_beams(txrx_pol[1])

    # echo ref and azimuth time
    ref_utc_echo, aztime_echo = _raw_obj.getPulseTimes('A', txrx_pol[0])

    az_time_mid = aztime_echo.mean()

    # az time in UTC string used for validation
    aztime_utc = (ref_utc_echo + TimeDelta(az_time_mid)).isoformat()

    # Parse echoes for all channels at once
    echo = _raw_obj.getRawDataset('A', txrx_pol)[:]
    num_channel, _, _, = echo.shape

    # check the number of echo channels v.s. that of antenna beams
    if (num_channel < 2):
        raise RuntimeError(
            f'Min two channel requires for "{txrx_pol}" products in L0B!')

    if (num_beam < num_channel):
        raise RuntimeError(f'Min {num_channel} beams is required in ANT file!')

    # generate unit-energy Hann-widnowed chirp for validation
    chirp_hann = _raw_obj.getChirp('A', txrx_pol[0])
    chirp_hann *= np.hanning(chirp_hann.size)
    chirp_hann /= np.linalg.norm(chirp_hann)

    # parse all EL cut patterns used for validation
    beam = _ant_obj.el_cut_all(txrx_pol[1])

    # get pos/vel/quaternion at aztime used for validation
    pos_ecef, vel_ecef = orbit.interpolate(az_time_mid)
    quat_ant2ecef = attitude.interpolate(az_time_mid)

    def _validate_ant_null_loc(self, beam_left, beam_right, az_ang,
                               el_ang_start, el_ang_step, ant_null_est,
                               echo_null_est, aztime_utc_est, conv_flags_est,
                               err_msg=''):
        """Function to validate Null products from antenna and echo

        Parameters
        ----------
        beam_left : np.ndarray(float)
            amplitude of EL-cut pattern of left beam in (linear)
        beam_right : np.ndarray(float)
            amplitude of EL-cut pattern of right beam in (linear)
        az_ang : float
            azimuth angle of EL-cut patterns in (rad)
        el_ang_start : float
            start EL angle in (rad)
        el_ang_step : float
            step EL angle in (rad)
        ant_null_est : isce3::antenna::NullProduct for antenna null with
            memebrs (slant_range, el_angle, doppler, magnitude)
        echo_null_est : isce3::antenna::NullProduct for echo null with
            members (slant_range, el_angle, doppler, magnitude)
        aztime_utc_est : isce3 DateTime object
            az time in UTC of the Null product
        conv_flags_est : isce3::antenna::NullConvergenceFlags with members
            (newton_solver, geometry_echo, geometry_antenna)
        err_msg : str, default=''

        """
        # const
        mdeg2rad = np.pi / 180e3

        # get location of peak
        idm_left = beam_left.argmax()
        idm_right = beam_right.argmax()
        idx_slice = slice(idm_left, idm_right)

        # limit the beams within peak-to-peak
        pow_left = beam_left[idx_slice]**2
        pow_right = beam_right[idx_slice]**2

        # form perfrect null pattern within coarse antenna el angle resolution
        pow_null = abs(pow_left - pow_right) / (pow_left + pow_right)

        # get min location within EL angle reslution
        idx_null = pow_null.argmin()
        mag_null = pow_null[idx_null]
        el_null = el_ang_start + (idx_null + idm_left) * el_ang_step

        el_ang_max_err = 0.5 * el_ang_step

        # get the null location and doppler in slant range
        sr_null_min, dop_null, _ = ant2rgdop(el_null - el_ang_max_err, az_ang,
                                             self.pos_ecef, self.vel_ecef,
                                             self.quat_ant2ecef,
                                             self.wavelength, self.dem_interp)

        sr_null_max, _, _ = ant2rgdop(el_null + el_ang_max_err, az_ang,
                                      self.pos_ecef, self.vel_ecef,
                                      self.quat_ant2ecef,
                                      self.wavelength, self.dem_interp)

        # validate antenna  null product
        npt.assert_(sr_null_min < ant_null_est.slant_range and
                    ant_null_est.slant_range < sr_null_max,
                    msg=f'Antenna Null Slant Range is wrong {err_msg}')

        npt.assert_allclose(ant_null_est.el_angle, el_null,
                            atol=el_ang_max_err,
                            err_msg=f'Antenna Null EL angle is wrong \
{err_msg}')

        npt.assert_allclose(ant_null_est.doppler, dop_null, atol=self.atol_dop,
                            err_msg=f'Antenna Null Doppler is wrong {err_msg}')

        npt.assert_array_less(ant_null_est.magnitude, mag_null,
                              err_msg=f'Antenna Null Mag is too \
large {err_msg}')

        # validate echo null product
        npt.assert_(sr_null_min < echo_null_est.slant_range and
                    echo_null_est.slant_range < sr_null_max,
                    msg=f'Echo Null Slant Range is wrong {err_msg}')

        el_err = abs(echo_null_est.el_angle - ant_null_est.el_angle)
        npt.assert_allclose(el_err, 0.0,
                            atol=self.atol_el_err_mdeg * mdeg2rad,
                            err_msg=f'Echo Null EL angle and that of antenna \
are not close enough {err_msg}')

        dop_err = abs(echo_null_est.doppler - ant_null_est.doppler)
        npt.assert_allclose(dop_err, 0.0, atol=self.atol_dop,
                            err_msg=f'Echo Null Doppler and that of antenna \
are not close enough {err_msg}')

        npt.assert_array_less(ant_null_est.magnitude, echo_null_est.magnitude,
                              err_msg=f'Echo Null Mag is too small {err_msg}')

        # validate azimuth time tag
        npt.assert_equal(aztime_utc_est.isoformat(), self.aztime_utc,
                         err_msg=f'Azimuth UTC time is incorrect {err_msg}')

        # validate convergence flags
        npt.assert_(conv_flags_est.newton_solver,
                    msg=f'newton_solver convergence flag is False {err_msg}')

        npt.assert_(conv_flags_est.geometry_echo,
                    msg=f'geometry_echo convergence flag is False {err_msg}')

        npt.assert_(conv_flags_est.geometry_antenna,
                    msg=f'geometry_antenna convergence flag is False \
{err_msg}')

    def test_constructor(self):
        el_null_obj = ElNullRangeEst(self.wavelength, self.sr_spacing,
                                     self.chirp_rate, self.chirp_dur,
                                     self.orbit, self.attitude)

        npt.assert_equal(el_null_obj.polyfit_deg, 6,
                         err_msg='Polyfit degree is wrong')

        npt.assert_equal(el_null_obj.grid_type_name, "EL_AND_AZ",
                         err_msg='Wrong grid type!')

        npt.assert_equal(el_null_obj.max_iter_null, 25,
                         err_msg='Wrong max number of iterations for\
Null location')

        npt.assert_allclose(el_null_obj.dem_ref_height, self._dem_ref,
                            atol=1e-8, err_msg='Wrong DEM Ref Height')

        npt.assert_equal(el_null_obj.max_iter_dem, 20,
                         err_msg='Wrong max number of iterations for DEM')

        npt.assert_allclose(el_null_obj.wave_length, self.wavelength,
                            err_msg='Wrong wavelength')

        npt.assert_allclose(el_null_obj.atol_dem, 1.0,
                            err_msg='Wrong abs tolerance for DEM')

        npt.assert_allclose(el_null_obj.atol_null, 1e-5,
                            err_msg='Wrong abs tolerance for Null location')

        npt.assert_allclose(el_null_obj.max_el_spacing,
                            0.5e-3 * np.pi / 180,
                            err_msg='Wrong max EL spacing')

        orbit_mid_tm = self.orbit.mid_datetime.seconds_of_day()
        npt.assert_allclose(el_null_obj.mid_time_orbit, orbit_mid_tm,
                            err_msg='Wrong orbit mid time in seconds')

        # check chirp ref within single-precision rel tolerance!
        npt.assert_allclose(el_null_obj.chirp_sample_ref, self.chirp_hann,
                            rtol=1e-6, err_msg='Wrong reference chirp!')

    def test_gen_null_range_doppler(self):
        # build the common object used for all beams/channels
        el_null_obj = ElNullRangeEst(self.wavelength, self.sr_spacing,
                                     self.chirp_rate, self.chirp_dur,
                                     self.orbit, self.attitude)

        # loop over all pair of channels/beams
        for nn in range(self.num_channel - 1):

            el_ang_step = np.diff(self.beam.angle[:2])[0]
            el_ang_start = self.beam.angle[0]
            az_ang = self.beam.cut_angle

            # estimate null locations in both Echo and Antenna domain
            tm_null, echo_null, ant_null, flag_null, pow_pat_null =\
                el_null_obj.genNullRangeDoppler(
                    self.echo[nn], self.echo[nn+1],
                    self.beam.copol_pattern[nn],
                    self.beam.copol_pattern[nn+1],
                    self.sr_start, el_ang_start,
                    el_ang_step, az_ang, self.az_time_mid)

            # validate null locations
            self._validate_ant_null_loc(
                abs(self.beam.copol_pattern[nn]),
                abs(self.beam.copol_pattern[nn+1]),
                az_ang, el_ang_start, el_ang_step,
                ant_null, echo_null, tm_null, flag_null,
                err_msg=f' for null # {nn} and {nn+1}')

            # check the size of null power patterns
            el_size = pow_pat_null.el.size
            npt.assert_equal(pow_pat_null.ant.size, el_size,
                             err_msg='Wrong antenna null pattern size for '
                             f'for null # {nn} and {nn+1}')
            npt.assert_equal(pow_pat_null.echo.size, el_size,
                             err_msg='Wrong echo null pattern size for '
                             f'for null # {nn} and {nn+1}')

            # Print results on screen per pair of beams/channels
            r2md = 180e3 / np.pi
            def amp2db(x): return 20*np.log10(x)
            print(f'*** Results for beams # ({nn+1},{nn+2}) ***')
            print(f' ANT Null Location (EL, SR) -> \
        ({np.rad2deg(ant_null.el_angle):.2f}, {ant_null.slant_range:.3f})\
        (deg, m)')
            print(f' Echo Null Location (EL, SR) -> \
        ({np.rad2deg(echo_null.el_angle):.2f}, {echo_null.slant_range:.3f})\
        (deg, m)')
            print(
                f' Newton_solver convergence flag ->\
{flag_null.newton_solver}')
            print(
                f' Geometry_echo convergence flag ->\
{flag_null.geometry_echo}')
            print(
                f' Geometry_antenna convergence flag ->\
{flag_null.geometry_antenna}')
            print(f' Doppler (ant, echo) -> \
({ant_null.doppler:.1f}, {echo_null.doppler:.1f}) (Hz, Hz)')
            mag_null_ant = amp2db(ant_null.magnitude)
            mag_null_echo = amp2db(echo_null.magnitude)
            print(f' Null Mag (ant, echo) -> \
({mag_null_ant:.1f},{mag_null_echo:.1f}) (dB, dB)')
            err_el = r2md * abs(echo_null.el_angle - ant_null.el_angle)
            print(f' Error in EL angle -> {err_el:.1f} (mdeg)')
            err_sr = abs(echo_null.slant_range - ant_null.slant_range)
            print(f' Error in Slant Range -> {err_sr:.1f} (m)')
