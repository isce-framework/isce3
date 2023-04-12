from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import scipy.special
from numpy.typing import ArrayLike

from isce3.ext.isce3.core import Poly2d


def restricted_weak_compositions(N: int, K: int) -> Iterator[tuple[int, ...]]:
    """
    Generate all possible K-restricted weak compositions of N.

    A :math:`K`-restricted weak composition of :math:`N` is an ordered sequence of
    :math:`K` nonnegative integers whose sum is :math:`N`.

    Parameters
    ----------
    N : int
        The integer to decompose (i.e. the sum of each sequence). Must be nonnegative.
    K : int
        The sequence length (i.e. the number of nonnegative integers whose sum is `N`).
        Must be greater than zero.

    Yields
    ------
    composition : tuple of int
        A sequence of `K` nonnegative integers whose sum is `N`.
    """
    if N < 0:
        raise ValueError("N must be a nonnegative integer")
    if K <= 0:
        raise ValueError("K must be > 0")

    # Use a recursive helper function.
    # Adapted from https://stackoverflow.com/a/59131521.
    def helper(n, k, sequence=()):
        if k == 0:
            if n == 0:
                yield sequence
        elif k == 1:
            if 0 <= n <= N:
                yield sequence + (n,)
        elif 0 <= n <= N * K:
            for m in reversed(range(N + 1)):
                yield from helper(n - m, k - 1, sequence + (m,))

    return helper(N, K)


def polynomial_combinations(inputs: ArrayLike, *, degree: int) -> np.ndarray:
    r"""
    Generate polynomial combinations of the inputs.

    Given a set of input variables, computes a set of output variables that are monomial
    terms of the inputs with degree less than or equal to `degree`. For example, the
    degree-2 polynomial combinations of :math:`x` and :math:`y` are

    .. math::

        \begin{bmatrix}
            1, &x, &y, &x^2, &xy, &y^2
        \end{bmatrix}

    Parameters
    ----------
    inputs : (M, N) array_like
        The input sample matrix. An MxN array with M variables and N samples per
        variable.
    degree : int
        The maximum degree of polynomial combinations. A nonnegative integer.

    Returns
    -------
    combinations : (K, N) numpy.ndarray
        A matrix of polynomial combinations of the input variables. An KxN array, where
        K is the number of possible polynomial combinations of the input variables of up
        to degree `degree` and N is the number of samples per variable.

    Notes
    -----
    The number of possible polynomial combinations of :math:`M` variables of up to
    degree :math:`D` is the number of :math:`(M+1)`-restricted `weak compositions
    <https://en.wikipedia.org/wiki/Composition_(combinatorics)>`_ of :math:`D`, which is
    :math:`K={M+D \choose D}`.
    """
    inputs = np.asanyarray(inputs)

    if inputs.ndim != 2:
        raise ValueError(
            "input array should be 2-dimensional (num variables x num samples)"
        )
    if degree < 0:
        raise ValueError("degree must be a nonnegative integer")

    # Get the total number of polynomial combinations of the inputs of up to the
    # specified degree.
    num_vars, num_samples = inputs.shape
    num_combinations = scipy.special.comb(num_vars + degree, degree, exact=True)

    # Initialize the output array.
    out = np.zeros_like(inputs, shape=(num_combinations, num_samples))

    # Get the exponent of each input variable for each possible polynomial combination
    # of the inputs. For example, if the input variables are [x, y, z], we should get an
    # iterator that yields a sequence of exponents [a, b, c, d] such that each
    # polynomial combination is given by (1^a * x^b * y^c * z^d).
    exponents = (
        np.asarray(e) for e in restricted_weak_compositions(degree, num_vars + 1)
    )

    # Generate polynomial combinations of the inputs.
    for i, exp in enumerate(exponents):
        out[i] = np.prod(np.power(inputs, exp[1:, None]), axis=0)

    # Sanity check that the length of the iterator matched the number of output array
    # rows.
    assert i + 1 == len(out)

    return out


def fit_bivariate_polynomial(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, *, degree: int
) -> Poly2d:
    """
    Fit a 2-D polynomial to the input data.

    Returns a bivariate polynomial that is the least squares fit to the data `z` sampled
    on the grid of points with coordinates `x` and `y`.

    Parameters
    ----------
    x : (M,) array_like
        x-coordinates of the sample points.
    y : (N,) array_like
        y-coordinates of the sample points.
    z : (N, M) array_like
        Values sampled on the grid of points defined by the `x` and `y` coordinates.
    degree : int
        The degree of the polynomial.

    Returns
    -------
    poly2d : isce3.core.Poly2d
        The bivariate polynomial resulting from least squares fitting to the data.
    """
    # Get the full grid of sample points.
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Flatten input arrays.
    xx = np.ravel(xx)
    yy = np.ravel(yy)
    z = np.ravel(z)

    # Get polynomial combinations of the grid coordinates.
    coords = np.stack([xx, yy], axis=0)
    combinations = polynomial_combinations(coords, degree=degree)

    # Estimate polynomial coefficients by performing linear least squares regression.
    coeffs, *_ = scipy.linalg.lstsq(combinations.T, z)

    # A `Poly2d` object stores a 2-D array of coefficients with shape
    # (y_order, x_order). In order to represent a bivariate polynomial with degree D as
    # a `Poly2d`, we need to create an array of coefficients with shape (D+1, D+1) and
    # only fill the upper-left triangular elements. The remaining values should be zero.
    n = degree + 1
    full_coeffs = np.zeros((n, n), dtype=np.float64)

    # Fill the array with polynomial coefficients.
    exponents = restricted_weak_compositions(degree, 3)
    for exp, coeff in zip(exponents, coeffs):
        idx = exp[1:][::-1]
        full_coeffs[idx] = coeff

    return Poly2d(full_coeffs)
