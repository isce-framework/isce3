#!/usr/bin/env python3
from nisar.products.readers.antenna import AntennaParser
from nisar.products.readers.antenna.antenna_parser import xdb_points_from_cut
from isce3.ext.isce3 import antenna as ant
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

    def test_beam_numbers(self):
        list_beams = [1, 2, 3, 4, 5]
        for pol in ['H', 'V']:
            npt.assert_equal(self.prs.beam_numbers(pol), list_beams,
                             err_msg="Wrong list of beam numbers for"
                             f" {pol} Pol")

    def test_el_cut(self):
        elcut = self.prs.el_cut(full=True)

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
        azcut = self.prs.az_cut(3, 'V', full=True)

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
        beam_first = self.prs.el_cut(1, full=True)
        beam_last = self.prs.el_cut(num_beam, full=True)
        ang_first = beam_first.angle[0]
        ang_last = beam_last.angle[-1]
        num_ang = int(np.ceil((ang_last - ang_first) /
                              (beam_first.angle[1] - beam_first.angle[0]))) + 1

        elcut = self.prs.el_cut_all(full=True)
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
                            err_msg="Wrong 'copol' for first beam")

        pattern = np.interp(angles, beam_last.angle, beam_last.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(elcut.copol_pattern[-1, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' for last beam")

    def test_cut_angles_az_cuts(self):
        num_beam = self.prs.num_beams()
        cut_angs = self.prs.cut_angles_az_cuts()
        npt.assert_equal(cut_angs.size, num_beam, err_msg="Wrong size!")

    def test_az_cut_all(self):
        num_beam = self.prs.num_beams()
        beam_first = self.prs.az_cut(1, full=True)
        beam_last = self.prs.az_cut(num_beam, full=True)
        ang_first = beam_first.angle[0]
        ang_last = beam_last.angle[-1]
        num_ang = int(np.ceil((ang_last - ang_first) /
                              (beam_first.angle[1] - beam_first.angle[0]))) + 1

        azcut = self.prs.az_cut_all(full=True)
        npt.assert_equal(azcut.copol_pattern.shape, (num_beam, num_ang),
                         err_msg="Wrong shape for 'copol'")

        angles = np.linspace(ang_first, ang_last, num_ang)
        npt.assert_allclose(azcut.angle, angles, atol=self.atol,
                            rtol=self.rtol, err_msg="Wrong 'angle'")

        cut_angs_az = self.prs.cut_angles_az_cuts()
        npt.assert_allclose(azcut.cut_angle, cut_angs_az.mean(), atol=0.01,
                            rtol=0.1, err_msg="Wrong mean 'cut_ang'")

        pattern = np.interp(angles, beam_first.angle, beam_first.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(azcut.copol_pattern[0, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' for first beam")

        pattern = np.interp(angles, beam_last.angle, beam_last.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(azcut.copol_pattern[-1, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' for last beam")

    def test_locate_beam_peak_overlap(self):
        for pol in ['H', 'V']:
            num_beams = self.prs.num_beams(pol)
            el_peaks, az_peaks = self.prs.locate_beams_peak(pol)
            # check the values are unique, ascending order and have right size
            el_sort = sorted(set(el_peaks))
            npt.assert_equal(el_sort, el_peaks,
                             err_msg='Non monotonically ascending peaks'
                             f' for pol "{pol}"!')
            # get EL angle for the first beam and last beam
            ant_first = self.prs.el_cut(1, pol)
            el_first = ant_first.angle[0]
            npt.assert_array_less(el_first, el_peaks,
                                  err_msg='One of more peaks are too small'
                                  f' for pol "{pol}"!')
            ant_last = self.prs.el_cut(num_beams, pol)
            el_last = ant_last.angle[-1]
            npt.assert_array_less(el_peaks, el_last,
                                  err_msg='One of more peaks are too large'
                                  f' for pol "{pol}"!')
            # check azimuth angle
            az_avg = 0.5 * (ant_first.cut_angle + ant_last.cut_angle)
            npt.assert_allclose(az_peaks, az_avg,
                                err_msg='Wrong azimuth for peaks'
                                f' for pol "{pol}"!')
            # now check overlap/transition regions agianst peak location
            el_trans, az_trans = self.prs.locate_beams_overlap(pol)
            npt.assert_equal(el_trans.size, num_beams - 1,
                             err_msg='Wrong size for overlap values'
                             f' for pol "{pol}"!')
            npt.assert_array_less(el_peaks[:-1], el_trans,
                                  err_msg='Wrong EL overlap values'
                                  f' for pol "{pol}"!')
            npt.assert_allclose(az_peaks, az_trans,
                                err_msg='Wrong azimuth for overlap values'
                                f' for pol "{pol}"!')

    def test_xdb_points_from_cut(self):
        cut = self.prs.el_cut()
        amp_pat = abs(cut.copol_pattern)
        idx_max = amp_pat.argmax()
        # get angular coverage for min dynamic range `xdb`
        xdb = 7.0
        ang_first, ang_last, slice_ang = xdb_points_from_cut(cut, x_db=xdb)
        npt.assert_(ang_first < cut.angle[idx_max],
                    msg='Wrong first/left angle value!')
        npt.assert_(ang_last > cut.angle[idx_max],
                    msg='Wrong last/right angle value!')
        npt.assert_(slice_ang.start < idx_max < slice_ang.stop,
                    msg='Wrong slice angle index!')
        # check the magnitude on both left and right side of the peak
        # to be at least "xdb" dB below the peak!
        pk2left = 20 * np.log10(
            amp_pat[idx_max] / amp_pat[slice_ang.start])
        npt.assert_(pk2left >= xdb,
                    msg=(f'Peak-to-Left mag is {pk2left:.2f} dB less than'
                         f' min {xdb} dB!')
                    )
        pk2right = 20 * np.log10(
            amp_pat[idx_max] / amp_pat[slice_ang.stop - 1])
        npt.assert_(pk2left >= xdb,
                    msg=(f'Peak-to-Right mag is {pk2right:.2f} dB less than'
                         f' min {xdb} dB!')
                    )

    def test_az_cut_all_truncated(self):
        num_beam = self.prs.num_beams()
        beam_first = self.prs.az_cut(1, full=False)
        beam_last = self.prs.az_cut(num_beam, full=False)
        ang_first = beam_first.angle[0]
        ang_last = beam_last.angle[-1]
        num_ang = int(np.ceil((ang_last - ang_first) /
                              (beam_first.angle[1] - beam_first.angle[0]))) + 1

        azcut = self.prs.az_cut_all(full=False)
        npt.assert_equal(azcut.copol_pattern.shape, (num_beam, num_ang),
                         err_msg="Wrong shape for 'copol'")

        angles = np.linspace(ang_first, ang_last, num_ang)
        npt.assert_allclose(azcut.angle, angles, atol=self.atol,
                            rtol=self.rtol, err_msg="Wrong 'angle'")

        cut_angs_az = self.prs.cut_angles_az_cuts()
        npt.assert_allclose(azcut.cut_angle, cut_angs_az.mean(), atol=0.01,
                            rtol=0.1, err_msg="Wrong mean 'cut_ang'")

        pattern = np.interp(angles, beam_first.angle, beam_first.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(azcut.copol_pattern[0, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' for first beam")

        pattern = np.interp(angles, beam_last.angle, beam_last.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(azcut.copol_pattern[-1, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' for last beam")

    def test_el_cut_all_truncated(self):
        num_beam = self.prs.num_beams()
        beam_first = self.prs.el_cut(1, full=False)
        beam_last = self.prs.el_cut(num_beam, full=False)
        ang_first = beam_first.angle[0]
        ang_last = beam_last.angle[-1]
        num_ang = int(np.ceil((ang_last - ang_first) /
                              (beam_first.angle[1] - beam_first.angle[0]))) + 1

        elcut = self.prs.el_cut_all(full=False)
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
                            err_msg="Wrong 'copol' for first beam")

        pattern = np.interp(angles, beam_last.angle, beam_last.copol_pattern,
                            left=0.0, right=0.0)
        npt.assert_allclose(elcut.copol_pattern[-1, :], pattern,
                            atol=self.atol, rtol=self.rtol,
                            err_msg="Wrong 'copol' for last beam")
