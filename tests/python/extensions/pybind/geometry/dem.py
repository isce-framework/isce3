#!/usr/bin/env python3
import pytest
import pybind_isce3.geometry as m
from pybind_isce3.core import DataInterpMethod

def test_const():
    href = 10.

    dem = m.DEMInterpolator()
    dem.ref_height = href
    assert dem.ref_height == href

    dem = m.DEMInterpolator(href)
    assert dem.ref_height == href

    assert dem.interpolate_xy(0, 0) == href
    assert dem.interpolate_lonlat(0, 0) == href

    assert dem.have_raster == False


def test_methods():
    # pybind11::enum_ is not iterable
    for name in "SINC BILINEAR BICUBIC NEAREST BIQUINTIC".split():
        # enum constructor
        method = getattr(DataInterpMethod, name)
        dem = m.DEMInterpolator(method=method)
        assert dem.interp_method == method
        # string constructor
        dem = m.DEMInterpolator(method=name)
        assert dem.interp_method == method

    dem = m.DEMInterpolator(method="bicubic")
    assert dem.interp_method == DataInterpMethod.BICUBIC

    dem = m.DEMInterpolator(method="biCUBic")
    assert dem.interp_method == DataInterpMethod.BICUBIC

    with pytest.raises(ValueError):
        dem = m.DEMInterpolator(method="TigerKing")

    # TODO Test other methods once we have isce::io::Raster bindings.
