#!/usr/bin/env python3
import pybind_isce3.geometry as m
from pybind_isce3.core import dataInterpMethod

def test_dem():
    href = 10.

    dem = m.DEMInterpolator()
    dem.refHeight = href
    assert dem.refHeight == href

    dem = m.DEMInterpolator(href)
    assert dem.refHeight == href

    assert dem.interpolateXY(0, 0) == href
    assert dem.interpolateLonLat(0, 0) == href

    method = dataInterpMethod.bicubic
    dem = m.DEMInterpolator(height=href, method=method)
    assert dem.interpMethod == method

    dem = m.DEMInterpolator(method="bicubic")
    assert dem.interpMethod == method

    assert dem.haveRaster == False

    # TODO Test other methods once we have isce::io::Raster bindings.
