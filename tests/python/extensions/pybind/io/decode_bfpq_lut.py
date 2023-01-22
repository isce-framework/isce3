import numpy as np
import numpy.testing as npt
import isce3.ext.isce3 as isce3

def test_decode_bfpq_lut():
    nlut = 2**8
    lut = np.random.normal(size=nlut).astype("f4")

    dtype = np.dtype([("r", np.uint16), ("i", np.uint16)])
    n = 32768
    encoded = np.zeros((1, n), dtype=dtype)
    encoded["r"] = np.random.randint(nlut, size=n)
    encoded["i"] = np.random.randint(nlut, size=n)

    decoded = lut[encoded["r"]] + 1j * lut[encoded["i"]]

    npt.assert_allclose(decoded, isce3.io.decode_bfpq_lut(lut, encoded))
