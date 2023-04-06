import numpy as np

import isce3
import isce3.ext.isce3.core as m


def test_constants():
    for method in "SINC BILINEAR BICUBIC NEAREST BIQUINTIC".split():
        assert hasattr(m.DataInterpMethod, method)
    assert hasattr(m, "earth_spin_rate")


def test_wgs84_ellipsoid():
    assert np.isclose(isce3.core.WGS84_ELLIPSOID.a, 6_378_137.0)
    assert np.isclose(isce3.core.WGS84_ELLIPSOID.b, 6_356_752.314_245)
