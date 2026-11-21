import numpy as np
import pytest

from isce3.atmosphere.split_band_estimation import (
    compute_unwrapp_error_main_diff_low_high,
    estimate_iono_main_diff_low_high)

F0 = 1_239e6
FL = 1_225_666_666.6666667
FH = 1_252_333_333.3333333
# finite-band approximation scale of the estimator
EPS = ((FH - FL) / (2 * F0)) ** 2


def _forward(n, i):
    """Noiseless unwrapped main and (high - low) phase."""
    def phase(f):
        return (f / F0) * n + (F0 / f) * i
    shape = (2, 3)
    return (np.full(shape, phase(F0)),
            np.full(shape, phase(FH) - phase(FL)))


@pytest.mark.parametrize("n, i", [(3.0, 0.0), (0.0, 2.0), (1.0, -0.5),
                                  (148.0, -153.0)])
def test_main_diff_low_high_recovers_components(n, i):
    main, diff = _forward(n, i)
    disp, non_disp = estimate_iono_main_diff_low_high(
        F0, FL, FH, main, diff)
    tol = 1e-9
    assert np.all(np.abs(disp - i) <= EPS * abs(i) + tol)
    assert np.all(np.abs(non_disp - n) <= EPS * abs(n) + tol)
    # main = non_dispersive + dispersive within the approximation
    assert np.all(np.abs(main - disp - non_disp)
                  <= EPS * (abs(n) + abs(i)) + tol)


@pytest.mark.parametrize("n, i", [(3.0, 0.0), (148.0, -153.0)])
def test_main_diff_low_high_no_spurious_unwrap_coefficients(n, i):
    main, diff = _forward(n, i)
    disp, non_disp = estimate_iono_main_diff_low_high(
        F0, FL, FH, main, diff)
    common, differential = compute_unwrapp_error_main_diff_low_high(
        F0, FL, FH, disp, non_disp, diff, main)
    assert not common.any()
    assert not differential.any()