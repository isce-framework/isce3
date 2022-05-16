#!/usr/bin/env python3

import isce3.ext.isce3.core as m

def test_constants():
    for method in "SINC BILINEAR BICUBIC NEAREST BIQUINTIC".split():
        assert hasattr(m.DataInterpMethod, method)
    assert hasattr(m, "earth_spin_rate")
