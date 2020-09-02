#!/usr/bin/env python3

import pybind_isce3.core as m

def test_constants():
    for method in "SINC BILINEAR BICUBIC NEAREST BIQUINTIC".split():
        assert hasattr(m.DataInterpMethod, method)
    assert hasattr(m, "earth_spin_rate")
