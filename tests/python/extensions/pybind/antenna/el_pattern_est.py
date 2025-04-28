#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt
import os
import bisect
from scipy.interpolate import interp1d

from isce3.ext.isce3.antenna import ElPatternEst
from nisar.products.readers.antenna.antenna_parser import AntennaParser
from nisar.products.readers.Raw import Raw
from isce3.ext.isce3.geometry import DEMInterpolator
import iscetest


# static method
def replace_nan_echo(echo, rnd_seed=10):
    """
    Replace NaN or Zero values by a Gaussian random noise
    whose STD determined by std of non-nan and non-zero values
    per range line

    Parameters
    ----------
        echo : np.ndarray(float)
        rnd_seed : int, default=10
            seed number for random generator
    Returns
    -------
        np.ndarray(float)
            Corrected echo with the same type and shape.

    """
    const_iq = 1. / np.sqrt(2.)
    # seed number for Gaussian noise random generator
    # to replace bad values of echo if any.
    rnd_gen = np.random.RandomState(rnd_seed)
    # get number of range lines and range bins
    nrgl, _ = echo.shape
    # replace bad values (NaN) with Gaussian noise with std determined
    # by non-nan range bins per range line
    for line in range(nrgl):
        mask_bad = np.isnan(echo[line]) | np.isclose(echo[line], 0)
        num_bad = mask_bad.sum()
        if num_bad > 0:
            std_iq = const_iq * np.std(echo[line, ~mask_bad])
            echo[line, mask_bad] = std_iq * (rnd_gen.randn(num_bad) +
                                             1j*rnd_gen.randn(num_bad))


# Test Fixture
class TestElPatternEst:
    # Tolerances in dB used for both MEAN and STD of residual Error

    # poly-fitted raw-echo-estimated pattern v.s. true
    # (knowledge of) antenna pattern used for final V&V. This is
    # looser requirement due to approximate antenna pattern, uncalibrated
    # instrument related stuff, range ambiguity, etc.
    atol_ant = 0.2
    # raw echo v.s. its poly fitted one, for intermdediate step
    # tighter requirement)
    atol_pf = 0.1

    # Antenna pattern file for ALOS1 PALSAR (approximate ones by ESA!)
    # Note that beam # 3 in this file is beam # 7 in ALOS1 PALSAR beam
    # numbering. The nominal look angle (peak) of beam # 7 per the
    # the reference stated below for "FBS343HH" shall be around 34.3 deg.
    # However, the peak location in the paper appears to be slightly at
    # smaller angle (Fig. 4.)! Thus, For FBD dataset used here we assume
    # a value within [34.0, 34.3]!
    # M. Shimada, et al, "PALSAR Radiometric and Geometric Calibration",
    # IEEE Trans. On GeoSci. & Remote Sensing, pp 3915-3931, December 2009.
    # See also H. Ghaemi , "EL Pointing Estimation of ALOS PALSAR"
    # https://github.jpl.nasa.gov/NISAR-POINTING/DOC.git
    _ant_file = 'ALOS1_PALSAR_ANTPAT_FIVE_BEAMS.h5'
    _lka_min_max = (34.0, 34.3)

    # filename of L0B ALOS1 PALSAR over Amazon Rainforest
    _filename = "ALPSRP081257070-H1.0__A_HH_2500_LINES.h5"
    _freq_band = 'A'
    _txrx_pol = 'HH'

    # desired slice of range lines
    # per experiments, min around 2500 pulses needed!
    _slice_rgl = slice(0, 2500)

    # assumed mean elevation height over entire Amazon
    # rain forest ~ 200.0 meters above sea level
    _mean_dem = 200.0

    # build DEM object for our target/scene
    dem_obj = DEMInterpolator(height=_mean_dem)

    # Following lines are parsed/obtained from L0B product
    _raw_obj = Raw(hdf5file=os.path.join(iscetest.data, _filename))

    # get orbit object
    orbit_obj = _raw_obj.getOrbit()

    # get azimuth mid time of the echo
    _, _tm_echo = _raw_obj.getPulseTimes(_freq_band, _txrx_pol[0])[_slice_rgl]
    az_tm_mid = _tm_echo.mean()

    # get chirp parameters
    _, _, chp_rate, chp_dur = _raw_obj.getChirpParameters(_freq_band,
                                                          _txrx_pol[0])

    # get slant range vector , start and spacing in meters
    _sr_lsp = _raw_obj.getRanges(_freq_band, _txrx_pol[0])
    sr_start = _sr_lsp.first
    sr_spacing = _sr_lsp.spacing

    # get decoded float raw echo data for desired range lines
    echo = _raw_obj.getRawDataset(_freq_band, _txrx_pol)[_slice_rgl]

    # replace bad values (NaN or zeros) with Gaussian noise in place
    replace_nan_echo(echo)

    def _parse_el_cut(self, beam=3, max_db=2.4, step_deg=0.02):
        """Prase one-way power pattern of EL cut with EL angles for
           a deisred beam over a max dynamic range. The pattern is
           interpolated to a finer angular resolution.

           Parameters
           ----------
           beam : int , default=3
           max_db : float , default=1.6
               max relative dynamic range of power pattern in dB
           step_deg : float, default=0.02
               Step size in new EL angle used in cubic interpolator

           Returns
           -------
           np.ndarray(float):
               EL angles in degrees
           np.ndarray(float):
               Peak-normalized EL-cut 1-way Power pattern in dB

        """
        el_cut = AntennaParser(os.path.join(iscetest.data, self._ant_file)
                               ).el_cut(beam=beam, pol=self._txrx_pol[0])
        # get peak normalized power pattern in dB
        pow_db = 20*np.log10(abs(el_cut.copol_pattern))
        idx_m = pow_db.argmax()
        pow_db -= pow_db[idx_m]
        # get indices for a max desired relative dynamic range [-max_db, 0]
        idx_l = bisect.bisect_left(pow_db[:idx_m], -max_db)
        idx_r = len(pow_db) - bisect.bisect_left(pow_db[:-idx_m], -max_db)
        el_deg = np.rad2deg(el_cut.angle[idx_l:idx_r])
        # interpolate power pattern to a finer uniform EL angles
        # using cubic spline 1-D interpolation
        f_intrp = interp1d(el_deg, pow_db[idx_l:idx_r], kind='cubic')
        el_deg_new = np.arange(el_deg[0], el_deg[-1], step_deg)
        pow_db_new = f_intrp(el_deg_new)
        pow_db_new -= pow_db_new.max()

        return el_deg_new, pow_db_new

    def test_constructors(self):
        # constructor # 1
        obj1 = ElPatternEst(self.sr_start, self.orbit_obj, polyfit_deg=3,
                            win_ped=0.5, center_scale_pf=True)

        npt.assert_equal(obj1.polyfit_deg, 3,
                         err_msg="Wrong polyfit order for 1st constructor!")

        npt.assert_allclose(obj1.win_ped, 0.5,
                            err_msg="Wrong raised-cosine window pedestal \
for 1st constructor!")

        npt.assert_allclose(obj1.is_center_scale_polyfit, True,
                            err_msg="Wrong 'center_scale_pf' flag for the \
1st constructor!")

        # constructor # 2
        obj2 = ElPatternEst(self.sr_start, self.orbit_obj,
                            dem_interp=self.dem_obj)

        npt.assert_equal(obj2.polyfit_deg, 6,
                         err_msg="Wrong polyfit order for 2ed constructor!")

        npt.assert_allclose(obj2.win_ped, 0.0,
                            err_msg="Wrong raised-cosine window pedestal \
for 2ed constructor!")

        npt.assert_allclose(obj2.is_center_scale_polyfit, False,
                            err_msg="Wrong 'center_scale_pf' flag for the \
2ed constructor!")

    def test_methods(self):
        # class constructor shared by all methods below
        el_pat_est = ElPatternEst(self.sr_start, self.orbit_obj,
                                  dem_interp=self.dem_obj)

        # parse 1-way antenna EL cut pattern for final V&V
        el_deg, ant_pat_el = self._parse_el_cut()

        # test 1-way power pattern method and perform V&V
        size_avg = 16
        p1w, slrg, lka_rad, inc_rad, pf_obj = el_pat_est.power_pattern_1way(
            self.echo, self.sr_spacing, self.chp_rate,
            self.chp_dur, self.az_tm_mid,
            size_avg=size_avg)
        len_p1w = p1w.size
        # check slant ranges
        npt.assert_equal(self.sr_start <= slrg.first, True,
                         err_msg='Wrong first slantrange for "pow_pat_1w"')
        npt.assert_allclose(slrg.spacing, self.sr_spacing * size_avg,
                            err_msg='Wrong slantrange spacing of "pow_pat_1w"')
        npt.assert_equal(slrg.size, len_p1w,
                         err_msg='Wrong slantrange size for "pow_pat_1w"')

        # check look angle
        npt.assert_equal(lka_rad.size, len_p1w,
                         err_msg='Wrong look angle size for "pow_pat_1w"')
        lka_deg = np.rad2deg(lka_rad)

        # check polyfit
        npt.assert_equal(pf_obj.order, 6,
                         err_msg='Poly-fit order must be 6 for "pow_pat_1w"')

        # check incidence angle to be monotnically increasing from look angles
        npt.assert_equal(inc_rad.size, len_p1w,
                         err_msg='Wrong incidence angle size for "pow_pat_1w"')
        npt.assert_array_less(lka_rad, inc_rad,
                              err_msg='All incidence angles must be greater \
than look angles for "pow_pat_1w"')

        # check the mean, std of diff between echo power and its polyfit
        pf_vals = pf_obj.eval(lka_rad)
        err = p1w - pf_vals  # (dB)
        npt.assert_allclose(err.mean(), 0.0, atol=self.atol_pf,
                            err_msg="Large mean error for poly\
fitted 1-way power pattern")
        npt.assert_allclose(err.std(), 0.0, atol=self.atol_pf,
                            err_msg="Large std error for poly\
fitted 1-way power pattern")

        # find the look angle at the peak (boresight angle)
        idx_max = np.argmax(pf_vals)
        lka_max_deg = np.round(lka_deg[idx_max], decimals=2)
        npt.assert_equal(lka_max_deg >= self._lka_min_max[0] and
                         lka_max_deg <= self._lka_min_max[1], True,
                         err_msg=f"Boresight angle is out of expected range \
{self._lka_min_max} (deg,deg)")

        # get polyfitted echo power within EL angle of antenna pattern
        # centered at estimated boresignth angle
        lka_ant_deg = lka_max_deg + el_deg
        pf_vals_ant = np.asarray(pf_obj.eval(np.deg2rad(lka_ant_deg)))
        pf_vals_ant -= pf_vals_ant.max()

        # diff between polyfitted echo and antenna one within antenna EL angles
        err = ant_pat_el - pf_vals_ant  # (dB)
        npt.assert_allclose(err.mean(), 0.0, atol=self.atol_ant,
                            err_msg="Large mean error for diff between poly\
fitted 1-way echo power pattern and antenna el cut")
        npt.assert_allclose(err.std(), 0.0, atol=self.atol_ant,
                            err_msg="Large std error for diff between poly\
fitted 1-way echo power pattern and antenna el cut")

        # test 2-way power pattern method
        p2w, slrg, lka_rad, inc_rad, pf_obj = el_pat_est.power_pattern_2way(
            self.echo, self.sr_spacing, self.chp_rate,
            self.chp_dur, self.az_tm_mid)
        len_p2w = p2w.size
        # check slant ranges
        npt.assert_equal(self.sr_start <= slrg.first, True,
                         err_msg='Wrong first slantrange for "pow_pat_2w"')
        npt.assert_allclose(slrg.spacing, self.sr_spacing * 8,
                            err_msg='Wrong slantrange spacing of "pow_pat_2w"')
        npt.assert_equal(slrg.size, len_p2w,
                         err_msg='Wrong slantrange size for "pow_pat_2w"')

        # check look angle
        npt.assert_equal(lka_rad.size, len_p2w,
                         err_msg='Wrong look angle size for "pow_pat_2w"')
        lka_deg = np.rad2deg(lka_rad)

        # check polyfit
        npt.assert_equal(pf_obj.order, 6,
                         err_msg='Poly-fit order must be 6 for "pow_pat_2w"')

        # check incidence angle to be monotnically increasing from look angles
        npt.assert_equal(inc_rad.size, len_p2w,
                         err_msg='Wrong incidence angle size for "pow_pat_2w"')
        npt.assert_array_less(lka_rad, inc_rad,
                              err_msg='All incidence angles must be greater \
than look angles for "pow_pat_2w"')
