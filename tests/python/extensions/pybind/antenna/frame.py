#!/usr/bin/env python3

import itertools as it
import numpy as np
import numpy.testing as npt

from isce3.ext.isce3 import antenna as ant


class TestConstructors:
    def test_default(self):
        frm = ant.Frame()

        npt.assert_equal(frm.grid_type, ant.SphGridType.EL_AND_AZ,
                         "Expect 'EL_AND_AZ' as a default grid")

    def test_string(self):
        frm = ant.Frame("THETA_PHI")

        npt.assert_equal(frm.grid_type, ant.SphGridType.THETA_PHI)

    def test_gridtype(self):
        frm = ant.Frame(ant.SphGridType.EL_OVER_AZ)

        npt.assert_equal(frm.grid_type, ant.SphGridType.EL_OVER_AZ)


class TestTransformations:
    # Tolerances
    _rtol = 1e-7
    _atol = 1e-8

    # EL/AZ Angles in (deg)
    _el_deg = [-7.0, -2.0, 0.0, 1.0, 4.0]
    _az_deg = [-0.9, -0.4, 0.0, 0.4, 0.9]

    # Estimated XYZ for EL_AND_AZ grid
    _est_xyz = np.zeros((len(_el_deg), 3))

    _est_xyz[0, :] = [-0.121864326797,
                      -0.015668270588,
                      0.992423090799]

    _est_xyz[1, :] = [-0.034899213187,
                      -0.006979842637,
                      0.999366462673]

    _est_xyz[2, :] = [0.000000000000,
                      0.000000000000,
                      1.000000000000]

    _est_xyz[3, :] = [0.017452264667,
                      0.006980905867,
                      0.999823327099]

    _est_xyz[4, :] = [0.069753604227,
                      0.015694560951,
                      0.997440782931]

    # Frame object in EL_AND_AZ
    _frame = ant.Frame()

    # EL/AZ Angles converted to radinas
    _el_rad = np.deg2rad(_el_deg)
    _az_rad = np.deg2rad(_az_deg)

    def test_scalars(self):
        err_msg_f = "'sph2cart' failed @ (el,az) = ({:.3f},{:.3f}) (deg,deg)"
        err_msg_b = "'cart2sph' failed @ (el,az) = ({:.3f},{:.3f}) (deg,deg)"

        for idx, (el, az, est_xyz) in enumerate(
                zip(self._el_rad, self._az_rad, self._est_xyz)):
            # Forward trasnformation
            v_xyz = self._frame.sph2cart(el, az)

            npt.assert_allclose(v_xyz, est_xyz, rtol=self._rtol,
                                atol=self._atol, err_msg=err_msg_f.format(
                                    self._el_deg[idx], self._az_deg[idx]))

            # Backward Transformation
            v_elaz = self._frame.cart2sph(v_xyz)

            npt.assert_allclose(v_elaz, [el, az], rtol=self._rtol,
                                atol=self._atol, err_msg=err_msg_b.format(
                                    self._el_deg[idx], self._az_deg[idx]))

    def test_vectors(self):
        # Forward transformation
        v_xyz = self._frame.sph2cart(self._el_rad, self._az_rad)

        npt.assert_allclose(v_xyz, self._est_xyz, atol=self._atol,
                            rtol=self._rtol, err_msg="'sph2cart' failed")

        # Backward transformation
        v_elaz = self._frame.cart2sph(v_xyz)
        pairs_elaz = list(zip(self._el_rad, self._az_rad))

        npt.assert_allclose(v_elaz, pairs_elaz, atol=self._atol,
                            rtol=self._rtol, err_msg="'cart2sph' failed")

    def test_vectorscalar(self):
        # EL cut
        for idx, az in enumerate(self._az_rad):
            # Forward-Backward transformation
            v_elaz = self._frame.cart2sph(
                self._frame.sph2cart(self._el_rad, az))

            pairs_elaz = list(it.zip_longest(self._el_rad, [az], fillvalue=az))

            npt.assert_allclose(v_elaz, pairs_elaz, atol=self._atol,
                                rtol=self._rtol,
                                err_msg="EL-cut failed @ {:.3f} (deg)".format(
                                    self._az_deg[idx]))

        # AZ cut
        for idx, el in enumerate(self._el_rad):
            # Forward-Backward transformation
            v_elaz = self._frame.cart2sph(
                self._frame.sph2cart(el, self._az_rad))

            pairs_elaz = list(it.zip_longest([el], self._az_rad, fillvalue=el))

            npt.assert_allclose(v_elaz, pairs_elaz, atol=self._atol,
                                rtol=self._rtol,
                                err_msg="AZ-cut failed @ {:.3f} (deg)".format(
                                    self._el_deg[idx]))


class TestOperators:
    _frame = ant.Frame()

    def test_equal(self):
        npt.assert_(self._frame == ant.Frame(ant.SphGridType.EL_AND_AZ))

    def test_notequal(self):
        npt.assert_(self._frame != ant.Frame(ant.SphGridType.THETA_PHI))
