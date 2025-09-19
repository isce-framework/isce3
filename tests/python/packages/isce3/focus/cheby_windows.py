import numpy as np
from isce3.focus import get_window_approximation
import pytest

n = 256
beta = 1.6
cases = (
    ("hamming", get_window_approximation("cosine", 0.08), np.hamming(n)),
    ("hann", get_window_approximation("cosine", 0.0), np.hanning(n)),
    ("kaiser", get_window_approximation("kaiser", beta), np.kaiser(n, beta)),
)

@pytest.mark.parametrize("name,cheby,expected", cases)
def test_window_approximation(name, cheby, expected):
    t = np.linspace(-0.5, 0.5, len(expected))
    values = np.array([cheby(ti) for ti in t])
    np.testing.assert_allclose(values, expected, atol=5e-6, err_msg=name)