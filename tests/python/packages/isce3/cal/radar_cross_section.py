import numpy as np
import numpy.testing as npt
import pytest
from numpy.random import default_rng

from isce3.cal.radar_cross_section import (
    estimate_peak_value,
    estimate_peak_width,
    zero_crossings,
)


class TestEstimatePeakValue:
    def test_random_quadratics(self):
        # Sample grid coordinates.
        x = np.linspace(-10.0, 10.0, 201)[None, :]
        y = np.linspace(-10.0, 10.0, 201)[:, None]

        # Randomly sample parameters of simple concave quadratic functions.
        rng = default_rng(seed=1234)
        n = 10
        x0s, y0s, z0s = (rng.uniform(-5.0, 5.0, size=n) for _ in range(3))
        cxs, cys = (rng.uniform(-2.5, -0.1, size=n) for _ in range(2))

        for x0, y0, z0, cx, cy in zip(x0s, y0s, z0s, cxs, cys):
            # Construct a 2-D array of samples of the quadratic function.
            z = z0 + cx * (x - x0) ** 2 + cy * (y - y0) ** 2

            # Estimate the peak and compare to the known true peak.
            peak = estimate_peak_value(z)
            assert np.isclose(peak, z0)

    @pytest.mark.parametrize("cx,cy", [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)])
    def test_concave_or_saddle_point(self, cx: float, cy: float):
        # Sample grid coordinates.
        x = np.linspace(-10.0, 10.0, 201)[None, :]
        y = np.linspace(-10.0, 10.0, 201)[:, None]

        # Construct a 2-D array of samples of the quadratic function that is concave or
        # contains a saddle point.
        z = cx * x ** 2 + cy * y ** 2

        # Trying to estimate the peak value should raise an exception.
        errmsg = "the input array did not contain a well-formed peak"
        with pytest.raises(RuntimeError, match=errmsg):
            estimate_peak_value(z)


class TestZeroCrossings:
    def test_line(self):
        # Estimate a single zero-crossing with sub-sample resolution.
        x = np.linspace(0.0, 10.0, 101)
        y = 3.0 * x - 0.15
        xz = zero_crossings(x, y)
        npt.assert_allclose(xz, [0.05])

    def test_sine(self):
        # Estimate multiple zero-crossings with sub-sample resolution.
        x = np.linspace(-10.0, 10.0, 1001)
        y = np.sin(x)
        xz = zero_crossings(x, y)
        expected = np.arange(-3, 4) * np.pi
        npt.assert_allclose(xz, expected)

    def test_no_crossings(self):
        # No zero-crossings occurred. Should return an empty array.
        x = np.linspace(-1.0, 1.0, 101)
        y = np.ones_like(x)
        xz = zero_crossings(x, y)
        assert len(xz) == 0

    def test_all_zeros(self):
        # If the `y` array is all zeros, the data is treated as containing no
        # zero-crossings.
        x = np.linspace(-1.0, 1.0, 101)
        y = np.zeros_like(x)
        xz = zero_crossings(x, y)
        assert len(xz) == 0

    def test_root_but_not_crossing(self):
        # If a (nonnegative-valued) function y(x) touches 0, but no sign change occurs,
        # this is not treated as a zero-crossing.
        x = np.linspace(-1.0, 1.0, 101)
        y = np.abs(x)
        xz = zero_crossings(x, y)
        assert len(xz) == 0

    def test_endpoints(self):
        # A zero-crossing can occur at the left and/or right endpoints.
        x = np.linspace(-1.0, 1.0, 101)
        y = x ** 2 - 1.0
        xz = zero_crossings(x, y)
        npt.assert_allclose(xz, [-1.0, 1.0])


class TestEstimatePeakWidth:
    def test_gaussian(self):
        # Sample points & sample spacing.
        x = np.linspace(-2.5, 2.5, 1001)
        dx = 0.005

        # Gaussian-shaped samples.
        y = np.exp(-0.5 * (x ** 2))

        # Get the 3dB width of the Gaussian.
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))

        # Estimate the peak width and compare with expected.
        width = dx * estimate_peak_width(y, threshold=0.5)
        assert np.isclose(width, fwhm)

    @pytest.mark.parametrize("threshold", [0.01, 0.1, 0.9, 0.99])
    def test_triang(self, threshold: float):
        # Sample points & sample spacing.
        x = np.linspace(-1.0, 1.0, 1001)
        dx = 0.002

        # Construct samples from a triangular-shaped function.
        y = 1.0 - 2.0 * np.abs(x)

        # Estimate the width at each threshold and compare with the expected width.
        width = dx * estimate_peak_width(y, threshold=threshold)
        assert np.isclose(width, 1.0 - threshold)

    def test_no_peak(self):
        # An exception should be raised if the input data never crosses the threshold
        y = np.zeros(100)
        errmsg = "the input array did not contain a well-formed peak"
        with pytest.raises(RuntimeError, match=errmsg):
            estimate_peak_width(y, threshold=0.5)

    def test_step_func(self):
        # ... or if it crosses exactly once
        x = np.linspace(-1.0, 1.0, 101)
        y = np.piecewise(x, [x > 0.0], [0.0, 1.0])
        errmsg = "the input array did not contain a well-formed peak"
        with pytest.raises(RuntimeError, match=errmsg):
            estimate_peak_width(y, threshold=0.5)

    def test_multiple_peaks(self):
        # ... or if it crosses more than twice
        x = np.linspace(-10.0, 10.0, 1001)
        y = np.sin(x)
        errmsg = "the input array contained multiple peaks"
        with pytest.raises(RuntimeError, match=errmsg):
            estimate_peak_width(y, threshold=0.0)
