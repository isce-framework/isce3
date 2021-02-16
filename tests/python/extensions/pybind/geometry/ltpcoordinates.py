#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt

from pybind_isce3 import geometry as geom


class TestLTPvectors:
    #  Tolerances
    _atol = 1e-7
    _rtol = 1e-6

    #  velocity (m/s)
    _sc_vel_ecf = [522.995925360679,
                   5107.808531616465,
                   5558.156209869601]

    # geodetic position (rad,rad,m)
    _sc_pos_llh = [np.deg2rad(-116.795192003152),
                   np.deg2rad(40.879509088888),
                   755431.529907600489]

    # expected values
    # heading in (deg)
    _est_heading = -14.0406212

    #  NED velocity vector in (m/s)
    _est_sc_vel_ned = [7340.716338644244,
                       -1835.775025245533,
                       -12.119371107411]

    def test_heading(self):
        hdg = geom.heading(*self._sc_pos_llh[:2], self._sc_vel_ecf)
        npt.assert_allclose(np.rad2deg(hdg), self._est_heading,
                            rtol=self._rtol, atol=self._atol)

    def test_ned_vector(self):
        v_ned = geom.ned_vector(*self._sc_pos_llh[:2], self._sc_vel_ecf)
        npt.assert_allclose(v_ned, self._est_sc_vel_ned, rtol=self._rtol,
                            atol=self._atol)

    def test_nwu_vector(self):
        v_nwu = geom.nwu_vector(*self._sc_pos_llh[:2], self._sc_vel_ecf)
        est_sc_vel_nwu = np.asarray(self._est_sc_vel_ned)
        est_sc_vel_nwu[1:] *= -1
        npt.assert_allclose(v_nwu, est_sc_vel_nwu, rtol=self._rtol,
                            atol=self._atol)

    def test_enu_vector(self):
        v_enu = geom.enu_vector(*self._sc_pos_llh[:2], self._sc_vel_ecf)
        est_sc_vel_enu = [self._est_sc_vel_ned[1], self._est_sc_vel_ned[0],
                          -self._est_sc_vel_ned[2]]
        npt.assert_allclose(v_enu, est_sc_vel_enu, rtol=self._rtol,
                            atol=self._atol)
