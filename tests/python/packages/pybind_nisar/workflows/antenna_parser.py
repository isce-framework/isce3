#!/usr/bin/env python3
from pybind_nisar.products.readers.antenna import AntennaParser
from pybind_isce3 import antenna as ant
import iscetest

import os
import numpy as np
import numpy.testing as npt


class TestAntennaParser:
    # name of HDF5 antenna pattern file
    filename = "ALOS1_PALSAR_ANTPAT_FIVE_BEAMS.h5"

    # antenna_parser object
    prs = AntennaParser(os.path.join(iscetest.data, filename))

    # tolerance
    atol = 1e-1
    rtol = 1e-2

    def test_filename(self):
        npt.assert_equal(self.prs.filename.split(os.sep)[-1], self.filename)

    def test_frame(self):
        npt.assert_equal(self.prs.frame, ant.Frame())

    def test_frequency(self):
        npt.assert_allclose(self.prs.frequency/1e9, 1.270)

    def test_num_beams(self):
        npt.assert_equal(self.prs.num_beams('H'), 5,
                         err_msg="Wrong number of beams for 'H' Pol")
        npt.assert_equal(self.prs.num_beams('V'), 5,
                         err_msg="Wrong number of beams for 'V' Pol")

    def test_timestamp(self):
        npt.assert_equal(self.prs.timestamp, "2021-03-04T19:33:29.238445")

    def test_rx_beams(self):
        lst_rx_beams = self.prs.rx_beams
        npt.assert_(("RX03H" in lst_rx_beams) and ("RX05V" in lst_rx_beams))

    def test_tx_beams(self):
        npt.assert_equal(len(self.prs.tx_beams), 0)

    def test_tag(self):
        npt.assert_equal(self.prs.tag, "ALOS1-PALSAR")

    def test_version(self):
        npt.assert_equal(self.prs.version, "v2.0")

    def test_el_cut(self):
        elcut = self.prs.el_cut()

        npt.assert_allclose(np.rad2deg(elcut.cut_angle), 0.0,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'cut_ang' value")

        npt.assert_allclose(np.rad2deg([elcut.angle[0], elcut.angle[-1]]),
                            [-9.2, -1.6], atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'angle' values")

        abs_val = abs(elcut.copol_pattern)
        npt.assert_allclose([abs_val.min(), abs_val.max()],
                            [21.8776, 60.9537],
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' values")

        abs_val = abs(elcut.cxpol_pattern)
        npt.assert_allclose([abs_val.min(), abs_val.max()],
                            [0.0, 0.0],
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'cxpol' values")

    def test_az_cut(self):
        azcut = self.prs.az_cut(3, 'V')

        npt.assert_allclose(np.rad2deg(azcut.cut_angle), 0.1,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'cut_ang' value")

        npt.assert_allclose(np.rad2deg([azcut.angle[0], azcut.angle[-1]]),
                            [-9.937, 9.712], atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'angle' values")

        abs_val = abs(azcut.copol_pattern)
        npt.assert_allclose([abs_val.min(), abs_val.max()],
                            [0.33106, 65.6145],
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' values")

        abs_val = abs(azcut.cxpol_pattern)
        npt.assert_allclose([abs_val.min(), abs_val.max()],
                            [0.0, 0.0],
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'cxpol' values")

    def test_el_cut_all(self):
        num_beam = self.prs.num_beams()
        beam_first = self.prs.el_cut(1)
        beam_last = self.prs.el_cut(num_beam)
        ang_first = beam_first.angle[0]
        ang_last = beam_last.angle[-1]
        num_ang = int(np.ceil((ang_last - ang_first) /
                              (beam_first.angle[1] - beam_first.angle[0]))) + 1

        elcut = self.prs.el_cut_all()
        npt.assert_equal(elcut.copol_pattern.shape, (num_beam, num_ang),
                         err_msg="Wrong shape for 'copol'")

        angles = np.linspace(ang_first, ang_last, num_ang)
        npt.assert_allclose(elcut.angle, angles, atol=self.atol,
                            rtol=self.rtol, err_msg="Wrong 'angle'")

        npt.assert_allclose(np.rad2deg(elcut.cut_angle), 0.0, atol=0.01,
                            rtol=0.1, err_msg="Wrong mean 'cut_ang'")

        pattern = np.interp(angles, beam_first.angle, beam_first.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(elcut.copol_pattern[0, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'cxpol' for first beam")

        pattern = np.interp(angles, beam_last.angle, beam_last.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(elcut.copol_pattern[-1, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'cxpol' for last beam")
