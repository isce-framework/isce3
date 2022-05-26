import numpy as np
from isce3.ext.isce3 import focus

def test_chirp():
    T = 1.0
    K = 0.0
    fs = 1.0
    chirp = focus.form_linear_chirp(K, T, fs)
    assert np.allclose(chirp, [1+0j])
