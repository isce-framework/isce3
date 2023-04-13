import numpy as np
import numpy.testing as npt
import scipy.special

import isce3
from isce3.core.poly2d import polynomial_combinations, restricted_weak_compositions


class TestRestrictedWeakCompositions:
    def test_n3_k2(self):
        # Get each ordered sequence of 2 numbers that sum to 3.
        n = 3
        k = 2
        compositions = list(restricted_weak_compositions(n, k))

        # Check the number of possible combinations.
        assert len(compositions) == scipy.special.comb(n + k - 1, n, exact=True)

        # Check unique combinations.
        assert set(compositions) == {
            (3, 0),
            (2, 1),
            (1, 2),
            (0, 3),
        }

    def test_n2_k3(self):
        # Get each ordered sequence of 3 numbers that sum to 2.
        n = 2
        k = 3
        compositions = list(restricted_weak_compositions(n, k))

        # Check the number of possible combinations.
        assert len(compositions) == scipy.special.comb(n + k - 1, n, exact=True)

        # Check unique combinations.
        assert set(compositions) == {
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
        }


class TestPolynomialCombinations:
    def test_2vars_degree3(self):
        # Input array is a column vector with 3 variables and a single observation per
        # variable.
        x = np.asarray([[2.0, 3.0, 5.0]]).T

        # Get polynomial combinations of `x` with degree <= 2.
        y = polynomial_combinations(x, degree=2)

        npt.assert_allclose(
            y,
            [
                [1.0],  # 1
                [2.0],  # x
                [3.0],  # y
                [5.0],  # z
                [4.0],  # x^2
                [6.0],  # xy
                [10.0],  # xz
                [9.0],  # y^2
                [15.0],  # yz
                [25.0],  # z^2
            ],
        )

    def test_3vars_degree2(self):
        # Input array is a column vector with 2 variables and a single observation per
        # variable.
        x = np.asarray([[2.0, 3.0]]).T

        # Get polynomial combinations of `x` with degree <= 3.
        y = polynomial_combinations(x, degree=3)

        npt.assert_allclose(
            y,
            [
                [1.0],  # 1
                [2.0],  # x
                [3.0],  # y
                [4.0],  # x^2
                [6.0],  # xy
                [9.0],  # y^2
                [8.0],  # x^3
                [12.0],  # x^2y
                [18.0],  # xy^2
                [27.0],  # y^3
            ],
        )


class TestFitBivariatePolynomial:
    def test_fit_bivariate_polynomial(self):
        def f(x, y):
            return 1.0 + 2.0 * x - 1.5 * x * y + 0.5 * y ** 2 + 0.1 * x ** 3

        x = np.linspace(-2.0, 2.0, 101)
        y = np.linspace(-2.5, 2.5, 51)

        # Evaluate `f()` on a grid of points defined by `x` and `y` coordinates.
        xx, yy = np.meshgrid(x, y, indexing="ij", sparse=True)
        z = f(xx, yy)

        # Fit a cubic polynomial to the data.
        poly2d = isce3.core.fit_bivariate_polynomial(x, y, z, degree=3)

        coeffs = np.asarray(
            [
                [1.0, 2.0, 0.0, 0.1],
                [0.0, -1.5, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        npt.assert_allclose(poly2d.coeffs, coeffs, atol=1e-12)
