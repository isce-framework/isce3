import numpy as np
import numpy.testing as npt
from isce3.signal import cola_windows

def test_cola_property():
    n = 1013  # prime for partial windows
    x = np.zeros(n)
    for block, window in cola_windows(n):
        x[block] += window
    npt.assert_allclose(x, np.ones(n))
