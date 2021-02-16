#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt

from pybind_isce3 import geometry as geom
from pybind_isce3.core import Ellipsoid as ellips


class TestIntersect:
    #  Tolerances
    _atol = 1e-4
    _rtol = 1e-3

    # Inputs for target and spaceracft positions:

    # spacecraft geodetic position (rad,rad,m)
    _sc_pos_llh = [np.deg2rad(-116.795),
                   np.deg2rad(40.879),
                   755431.53]

    # target geodetic location on Ellipsoid with DEM height
    # in (rad,rad,m)
    _tg_pos_llh = [np.deg2rad(-123.393),
                   np.deg2rad(39.276),
                   800.0]

    # calculated common values from inputs
    _elp = ellips()
    _sc_pos_ecef = _elp.lon_lat_to_xyz(_sc_pos_llh)

    def test_slantrange_from_lookvec(self):
        # calculate some inputs plus expected values
        pnt_ecef = (self._elp.lon_lat_to_xyz(
            [self._tg_pos_llh[0], self._tg_pos_llh[1], 0.0])
            - self._sc_pos_ecef)
        est_slrg_wgs84 = np.linalg.norm(pnt_ecef)
        pnt_ecef /= est_slrg_wgs84

        # estimates from function under test
        sr = geom.slantrange_from_lookvec(self._sc_pos_ecef,
                                          pnt_ecef)

        npt.assert_allclose(sr, est_slrg_wgs84,
                            rtol=self._rtol, atol=self._atol)

        # test exception for bad slant range
        with npt.assert_raises(RuntimeError):
            sr = geom.slantrange_from_lookvec(self._sc_pos_ecef,
                                              -1*pnt_ecef)

        # test exception for bad input value "zero look vector"
        with npt.assert_raises(ValueError):
            sr = geom.slantrange_from_lookvec(self._sc_pos_ecef,
                                              0.0*pnt_ecef)

    def test_sr_pos_from_lookvec_dem(self):
        # calculate some inputs plus expected values
        est_loc_ecef = self._elp.lon_lat_to_xyz(self._tg_pos_llh)
        pnt_ecef_dem = est_loc_ecef - self._sc_pos_ecef
        est_slrg_dem = np.linalg.norm(pnt_ecef_dem)
        pnt_ecef_dem /= est_slrg_dem

        # estimates from function under test
        out = geom.sr_pos_from_lookvec_dem(self._sc_pos_ecef,
                                           pnt_ecef_dem,
                                           self._tg_pos_llh[-1])

        npt.assert_array_less(out["iter_info"], (10, 0.5))

        npt.assert_allclose(out["slantrange"], est_slrg_dem,
                            rtol=self._rtol, atol=self._atol)

        npt.assert_allclose(out["pos_xyz"], est_loc_ecef,
                            rtol=self._rtol, atol=self._atol)

        # Lon/lat in degrees
        npt.assert_allclose(np.rad2deg(out["pos_llh"][:2]),
                            np.rad2deg(self._tg_pos_llh[:2]),
                            rtol=self._rtol, atol=self._atol)
        # height within default accuracy = 0.5
        npt.assert_allclose(out["pos_llh"][-1],
                            self._tg_pos_llh[-1],
                            rtol=0.5, atol=0.5)

        # test exception for bad input for iteration params
        with npt.assert_raises(ValueError):
            out = geom.sr_pos_from_lookvec_dem(self._sc_pos_ecef,
                                               pnt_ecef_dem,
                                               self._tg_pos_llh[-1],
                                               num_iter=0)

        # test exception for bad input value "zero look vector"
        with npt.assert_raises(ValueError):
            out = geom.sr_pos_from_lookvec_dem(self._sc_pos_ecef,
                                               0.0*pnt_ecef_dem,
                                               self._tg_pos_llh[-1])
        # test exception for bad slant range
        with npt.assert_raises(RuntimeError):
            out = geom.sr_pos_from_lookvec_dem(self._sc_pos_ecef,
                                               -1*pnt_ecef_dem,
                                               self._tg_pos_llh[-1])
