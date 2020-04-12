#!/usr/bin/env python3
import pytest
import pybind_isce3.geometry as m
from pybind_isce3.core import DataInterpMethod

def test_const():
    href = 10.

    dem = m.DEMInterpolator()
    dem.refHeight = href
    assert dem.refHeight == href

    dem = m.DEMInterpolator(href)
    assert dem.refHeight == href

    assert dem.interpolateXY(0, 0) == href
    assert dem.interpolateLonLat(0, 0) == href

    assert dem.haveRaster == False


def test_methods():
    # pybind11::enum_ is not iterable
    for name in "SINC BILINEAR BICUBIC NEAREST BIQUINTIC".split():
        # enum constructor
        method = getattr(DataInterpMethod, name)
        dem = m.DEMInterpolator(method=method)
        assert dem.interpMethod == method
        # string constructor
        dem = m.DEMInterpolator(method=name)
        assert dem.interpMethod == method

    dem = m.DEMInterpolator(method="bicubic")
    assert dem.interpMethod == DataInterpMethod.BICUBIC

    dem = m.DEMInterpolator(method="biCUBic")
    assert dem.interpMethod == DataInterpMethod.BICUBIC

    with pytest.raises(ValueError):
        dem = m.DEMInterpolator(method="TigerKing")

    # TODO Test other methods once we have isce::io::Raster bindings.
