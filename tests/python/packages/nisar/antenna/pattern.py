from isce3.geometry import DEMInterpolator
import isce3
import iscetest
import math
from nisar.antenna.pattern import (TimingFinder, AntennaPattern, AntennaParser,
    find_changes)
from nisar.products.readers.Raw import Raw
from nisar.products.readers.instrument import InstrumentParser
from nisar.workflows.focus import make_doppler_lut
from isce3.focus import make_el_lut
import numpy as np
import numpy.testing as npt
from pathlib import Path


def _expect_equal_at_index(i, rd_all, wd_all, wl_all, rd, wd, wl):
    npt.assert_array_equal(rd, rd_all[i])
    npt.assert_array_equal(wd, wd_all[i])
    npt.assert_array_equal(wl, wl_all[i])


def test_find_changes():
    x = np.ones(10)
    changes = find_changes(x)
    assert len(changes) == 0


def test_timing():
    rd = np.zeros((100, 12), dtype='uint32')
    wd = np.zeros_like(rd)
    wl = np.zeros_like(wd)
    t = np.arange(rd.shape[0], dtype=float)

    # no changes
    finder = TimingFinder(t, rd, wd, wl)
    i = 0
    _expect_equal_at_index(i, rd, wd, wl, *finder.get_dbf_timing(t[i]))

    # change at t=i=1
    rd[1, 0] = 1
    finder = TimingFinder(t, rd, wd, wl)
    _expect_equal_at_index(0, rd, wd, wl, *finder.get_dbf_timing(t[0]))
    _expect_equal_at_index(1, rd, wd, wl, *finder.get_dbf_timing(t[1]))
    _expect_equal_at_index(1, rd, wd, wl, *finder.get_dbf_timing(t[1] + 0.4))
    _expect_equal_at_index(1, rd, wd, wl, *finder.get_dbf_timing(t[1] + 0.9))


def test_pattern():
    fn_raw = Path(iscetest.data) / "bf" / "REE_L0B_ECHO_DATA.h5"
    fn_ant = Path(iscetest.data) / "bf" / "REE_ANTPAT_CUTS_DATA.h5"
    fn_ins = Path(iscetest.data) / "bf" / "REE_INSTRUMENT_TABLE.h5"
    raw = Raw(hdf5file=fn_raw)
    dem = DEMInterpolator()
    ant = AntennaParser(fn_ant)
    ins = InstrumentParser(fn_ins)
    orbit = raw.getOrbit()
    attitude = raw.getAttitude()
    pol = "HH"

    fc, dop = make_doppler_lut([fn_raw], 0, orbit, attitude, dem)
    wavelength = isce3.core.speed_of_light / fc
    rdr2geo_params = dict(
        tol_height = 1e-5,
        look_min = 0,
        look_max = math.pi / 2,
    )
    el_lut = make_el_lut(orbit, attitude,
                         raw.identification.lookDirection,
                         dop, wavelength, dem,
                         rdr2geo_params)

    for lut in (None, el_lut):
        ap = AntennaPattern(raw, dem, ant, ins, orbit, attitude, el_lut=lut)
        epoch, times = raw.getPulseTimes()
        r = raw.getRanges("A", tx=pol[0])
        z = ap.form_pattern(times[0], r)

        npt.assert_(z[pol].shape == (len(r),))
        npt.assert_(z[pol].dtype == np.complex64)
