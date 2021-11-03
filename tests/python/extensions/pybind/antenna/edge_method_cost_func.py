#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt

from pybind_isce3 import antenna as ant
from pybind_isce3.core import Poly1d
from pybind_isce3.core import Linspace

# functions for conversion from (rad) to (mdeg) and vice versa


def r2md(ang_rad): return 1000 * np.rad2deg(ang_rad)
def md2r(ang_mdeg): return 1e-3 * np.deg2rad(ang_mdeg)


class TestEdgeMethodCostFunc:
    # List of input parameters:
    # max absolute pointing error (mdeg)
    max_abs_err_mdeg = 5.0

    # Absolute function value tolerance and max itereration
    # in root of cost function
    abs_tol = 1e-4
    max_iter = 20

    # look angle (off-nadir angle) inputs
    min_lka_edge_deg = 32.8
    max_lka_edge_deg = 34.0
    prec_lka_edge_deg = 1e-3

    # gain offset in (dB) between relative EL power patterns extracted from
    # antenna and echo. the roll offset estimation is insensitive to this gain
    # offset!
    gain_ofs = 0.5

    # desired roll angle offset in (mdeg) , ground truth values used for V&V
    roll_ofs_ant_mdeg = [-198.0, -42.5, 0.0, 67.0, 157.3]

    # Build a 6-order polyminals of a relative antenna gain from gain (dB)
    # versus look angles (rad) to be used as a reference for building both
    # antenna and echo data These points are extracted from a realitic
    # EL power pattern of ALOS1 beam 7.
    gain = [-2.2, -1.2, -0.55, -0.2, 0.0, -0.2, -0.5, -1.0, -2.0]
    lka_deg = [32.0, 32.5, 33.0, 33.5, 34.1, 34.5, 35., 35.5, 36.]

    # Calculated parameters from input parameters:
    num_lka_edge = int((max_lka_edge_deg - min_lka_edge_deg) /
                       prec_lka_edge_deg) + 1

    # form gain and look angle of reference antenna pattern, perform
    # polyfiting to build Poly1d object version of reference antenna pattern
    pf_ref = Poly1d(np.polyfit(np.deg2rad(lka_deg), gain, 6)[::-1])

    # uniformly-spaced look angles around rising edge used for both antenna
    # and echo objects
    lka_edge_rad = np.linspace(np.deg2rad(min_lka_edge_deg),
                               np.deg2rad(max_lka_edge_deg),
                               num_lka_edge)

    # form ANT 3rd-order poly object with roll offset applied to edge
    # look angles
    pf_ant_vec = []
    for roll in roll_ofs_ant_mdeg:
        lka_ant_rad = lka_edge_rad + md2r(roll)
        gain_ant = np.polyval(pf_ref[::-1], lka_ant_rad) + gain_ofs
        pf_ant_vec.append(Poly1d(np.polyfit(lka_edge_rad, gain_ant, 3)[::-1]))

    # form echo 3rd-order poly object with roll offset applied to edge look
    # angles
    gain_echo = np.polyval(pf_ref[::-1], lka_edge_rad)
    pf_echo = Poly1d(np.polyfit(lka_edge_rad, gain_echo, 3)[::-1])

    # here we use constant but non-normalized weights (order 0)
    pf_wgt = Poly1d([10], 0.0, 1.0)

    def _validate_estimation(self, est: tuple, roll_true_mdeg: float,
                             err_msg: str = ""):
        roll_est, f_val, flag, n_iter = est
        # abs error in (mdeg)
        abs_err = np.abs(r2md(roll_est) + roll_true_mdeg)
        # error msg
        err_msg1 = f'@true roll offset {roll_true_mdeg:.6f} (mdeg) {err_msg}'
        # validate
        npt.assert_(n_iter <= self.max_iter,
                    msg="Exceed max number of iteration " + err_msg1)
        npt.assert_(flag, msg="Wrong convergence flag " + err_msg1)
        npt.assert_allclose(abs_err, 0.0, atol=self.max_abs_err_mdeg,
                            err_msg=("Too large residual roll offset " +
                                     err_msg1))
        npt.assert_allclose(f_val, 0.0, atol=self.abs_tol,
                            err_msg="Wrong cost function value " + err_msg1)

    def test_roll_angle_offset_from_edge_scalars(self):
        # loop over roll offset (and antenna polyfit objects)
        for pf_ant, roll_true_mdeg in zip(self.pf_ant_vec,
                                          self.roll_ofs_ant_mdeg):

            # estimate roll angle offset w/o weighting
            est_tuple = ant.roll_angle_offset_from_edge(
                self.pf_echo, pf_ant,
                np.deg2rad(self.min_lka_edge_deg),
                np.deg2rad(self.max_lka_edge_deg),
                np.deg2rad(self.prec_lka_edge_deg))
            # validate
            self._validate_estimation(est_tuple, roll_true_mdeg,
                                      err_msg="w/o weighting")

            # estimate roll angle offset w/ weighting
            est_wgt_tuple = ant.roll_angle_offset_from_edge(
                self.pf_echo, pf_ant,
                np.deg2rad(self.min_lka_edge_deg),
                np.deg2rad(self.max_lka_edge_deg),
                np.deg2rad(self.prec_lka_edge_deg),
                self.pf_wgt)

            # validate
            self._validate_estimation(est_wgt_tuple, roll_true_mdeg,
                                      err_msg="w/ weighting")

    def test_roll_angle_offset_from_edge_linspace(self):
        lka_lsp = Linspace(np.deg2rad(self.min_lka_edge_deg),
                           np.deg2rad(self.prec_lka_edge_deg),
                           self.num_lka_edge)

        for pf_ant, roll_true_mdeg in zip(self.pf_ant_vec,
                                          self.roll_ofs_ant_mdeg):
            # estimate roll angle offset w/o weighting
            est_tuple = ant.roll_angle_offset_from_edge(
                self.pf_echo, pf_ant, lka_lsp)
            # validate
            self._validate_estimation(est_tuple, roll_true_mdeg,
                                      err_msg="w/o weighting")

            # estimate roll angle offset w/ weighting
            est_wgt_tuple = ant.roll_angle_offset_from_edge(
                self.pf_echo, pf_ant, lka_lsp, self.pf_wgt)
            # validate
            self._validate_estimation(est_wgt_tuple, roll_true_mdeg,
                                      err_msg="w/ weighting")
