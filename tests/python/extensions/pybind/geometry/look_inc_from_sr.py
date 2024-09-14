#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt
import os

import isce3
from nisar.products.readers.Raw import open_rrsd
from isce3.ext.isce3 import geometry as geom
from isce3.ext.isce3.core import Ellipsoid
from isce3.ext.isce3.geometry import DEMInterpolator
from isce3.geometry import get_near_and_far_range_incidence_angles
from nisar.products.readers import SLC
import iscetest


def test_get_near_and_far_range_incidence_angles():
    '''
    Test computation of the near and far range incidence angles
    using radar grid from envisat.h5
    '''
    input_h5_path = os.path.join(iscetest.data, "envisat.h5")

    radar_grid = isce3.product.RadarGridParameters(input_h5_path)

    # init SLC object and extract necessary test params from it
    rslc = SLC(hdf5file=input_h5_path)
    orbit = rslc.getOrbit()

    near_range_inc_angle_rad, far_range_inc_angle_rad = \
        get_near_and_far_range_incidence_angles(radar_grid, orbit)

    near_range_inc_angle_deg = np.rad2deg(near_range_inc_angle_rad)
    far_range_inc_angle_deg = np.rad2deg(far_range_inc_angle_rad)

    assert np.isclose(near_range_inc_angle_deg, 18.7004, atol=1e-3)
    assert np.isclose(far_range_inc_angle_deg, 19.5825, atol=1e-3)


class TestLookIncAngles:
    # Tolerances in angles in degrees
    atol = 1e-1
    rtol = 1e-3

    # filename of L0B
    _filename = "REE_L0B_out17.h5"

    # Following data are parsed from L0B "REE_L0B_out17.h5"
    # parse L0B
    _raw_obj = open_rrsd(os.path.join(iscetest.data, _filename))

    # get orbit object
    orbit_obj = _raw_obj.getOrbit()

    # get slant range vector
    sr_vec = np.array(_raw_obj.getRanges())

    # get azimuth mid time of the echo
    _, tm_echo = _raw_obj.getPulseTimes()
    az_tm_mid = tm_echo.mean()

    # Following data are grabbed from config file "REE_L0B_out17.rdf"
    # form Ellipsoid object from PLANET paramteres
    ellips_obj = Ellipsoid(6378137.0, 669437.999014132e-8)

    # form DEMInterpolator object from TARGET parameters
    dem_obj = DEMInterpolator(height=0.0)

    # Start and mid look angles from RADAR and S/C parameters
    # which are used in validation process
    # mechanical boresight look angle
    lka_mb_deg = 37.0
    # DWP (starting) look angle
    lka_dwp_deg = 36.3

    def _est_inc_angle(self, sr, lka):
        r"""Estimate incidence angle from slant range and look angle
           via Law of Sines [1]_, the incidence angle :math:`\theta_{i}` is
           .. math::
               \theta_{i} = \theta_{l} + sin^{-1}(sin(\theta_l)\frac{r}{R+h})
                where :math:`\theta_{l}` is look angle, 'r' is slant range,
                `R` is along-track range curvature and `h` is DEM height.

            Parameters
            ----------
                sr : float
                    Slant range in (m)
                lka : float
                    look angle in (rad)

            Returns
            -------
                float :
                    incidecne angle in (deg)

            References
            ----------
            ..[1] https://en.wikipedia.org/wiki/Law_of_sines
        """
        # get S/C height at mid azimuth time of echo
        pos_sc, vel_sc = self.orbit_obj.interpolate(self.az_tm_mid)

        # convert from ECEF to LLH of S/C position
        llh_sc = self.ellips_obj.xyz_to_lon_lat(pos_sc)

        # get heading of S/C
        hdg_sc = geom.heading(*llh_sc[:2], vel_sc)

        # get range curvature in along track direction
        rg_curv = self.ellips_obj.r_dir(hdg_sc, llh_sc[1])

        # get a ref DEM height
        # Note the DEM variation, local slope is ignored!
        dem_hgt = self.dem_obj.ref_height

        # calculate the incidence angle per Law of Sines
        inca = lka + np.arcsin(np.sin(lka) * sr / (rg_curv + dem_hgt))

        return np.rad2deg(inca)

    def test_look_inc_ang_from_sr_scalar(self):
        # get min look angles and incidence angles and valdiate them
        lka_min, inca_min = geom.look_inc_ang_from_slant_range(
            self.sr_vec[0], self.orbit_obj, self.az_tm_mid, self.dem_obj,
            self.ellips_obj)

        lka_min_deg = np.rad2deg(lka_min)
        inca_min_deg = np.rad2deg(inca_min)

        npt.assert_allclose(lka_min_deg, self.lka_dwp_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong min look angle")

        # get true incidence angle for a given look angle and slant range
        true_inca_deg = self._est_inc_angle(self.sr_vec[0], lka_min)
        npt.assert_allclose(inca_min_deg, true_inca_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong min incidnece angle")

        # get max look angles and incidence angles and valdiate them
        lka_max, inca_max = geom.look_inc_ang_from_slant_range(
            self.sr_vec[-1], self.orbit_obj, self.az_tm_mid, self.dem_obj,
            self.ellips_obj)

        lka_max_deg = np.rad2deg(lka_max)
        inca_max_deg = np.rad2deg(inca_max)

        # get look angle at around mid-swath in terms of EL angle coverage
        lka_mid_deg = 0.5 * (lka_max_deg + lka_min_deg)

        npt.assert_allclose(lka_mid_deg, self.lka_mb_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong max look angle")

        # get true incidence angle for a given look angle and slant range
        true_inca_deg = self._est_inc_angle(self.sr_vec[0], lka_max)
        npt.assert_allclose(inca_max_deg, true_inca_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong max incidnece angle")

    def test_look_inc_ang_from_sr_vector(self):
        lka_all, inca_all = geom.look_inc_ang_from_slant_range(
            self.sr_vec, self.orbit_obj, self.az_tm_mid, self.dem_obj,
            self.ellips_obj)

        len_sr = self.sr_vec.size
        npt.assert_equal(lka_all.size, len_sr,
                         err_msg="Wrong size for look angle vector")

        npt.assert_equal(inca_all.size, len_sr,
                         err_msg="Wrong size for incidence angle vector")

        # check look angle values
        lka_all_deg = np.rad2deg(lka_all)

        npt.assert_allclose(lka_all_deg[0], self.lka_dwp_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong first value in look angle vector")

        npt.assert_array_less(lka_all_deg[0], lka_all_deg[-1],
                              err_msg="Wrong last value in look angle vector")

        npt.assert_allclose(lka_all_deg.mean(), self.lka_mb_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong mean value in look angle vector")

        # check incidence angle values
        inca_all_deg = np.rad2deg(inca_all)

        true_inca_deg = self._est_inc_angle(self.sr_vec[0], lka_all[0])

        npt.assert_allclose(inca_all_deg[0], true_inca_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong first value in incidence vector")

        true_inca_deg = self._est_inc_angle(self.sr_vec[-1], lka_all[-1])
        npt.assert_allclose(inca_all_deg[-1], true_inca_deg,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong last value in incidence vector")
