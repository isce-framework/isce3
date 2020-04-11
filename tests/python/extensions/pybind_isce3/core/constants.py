#!/usr/bin/env python3

import pybind_isce3.core as m

def test_constants():
    for method in "sinc bilinear bicubic nearest biquintic".split():
        assert hasattr(m.dataInterpMethod, method)
