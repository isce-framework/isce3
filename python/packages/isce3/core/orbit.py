from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from isce3.core import Orbit


class OrbitPassDirection(str, Enum):
    """The pass direction of a satellite in polar orbit."""

    ASCENDING = "ascending"
    """An ascending pass, indicating a northward-traveling satellite."""
    DESCENDING = "descending"
    """A descending pass, indicating a southward-traveling satellite."""

    def __str__(self) -> str:
        return self.value


def count_sign_changes(x: ArrayLike) -> int:
    """
    Count the number of sign changes between consecutive elements in an array.

    A sign change occurs between two values if one is negative and the other is
    nonnegative. Zeros are considered nonnegative unless their sign bit is set.

    If the array is not 1-dimensional, the number of sign changes in the flattened array
    will be counted.

    An empty array has no sign changes.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    int
        The number of sign changes.
    """
    # Get a mask of negative-valued elements in `x`.
    is_negative = np.signbit(x)

    # Compute a pairwise logical XOR between adjacent elements in the mask. The result
    # is a boolean array where `True` values indicate that a sign changed occurred
    # between the corresponding element and its right neighbor in `x`.
    sign_changed = np.logical_xor(is_negative.flat[1:], is_negative.flat[:-1])

    # Get the total number of sign changes in the array.
    return np.sum(sign_changed)


class AmbiguousOrbitPassDirection(Exception):
    """Raised when the pass direction of an orbit could not be clearly discerned."""


def get_orbit_pass_direction(orbit: Orbit) -> OrbitPassDirection:
    """
    Get the pass direction (ascending or descending) of an orbit interval.

    In the event that the input orbit contains more than one pass direction (e.g. it
    crosses near the poles) the pass direction of the majority of the orbit's time tags
    is returned.

    If the orbit pass direction changes multiple times, then no clear pass direction can
    be discerned and an `AmbiguousOrbitPassDirection` exception will be raised. Consider
    cropping the orbit sampling interval to a narrower interval in this case.

    Parameters
    ----------
    orbit : Orbit
        The input orbit. Must contain at least one state vector. Orbit state vectors are
        assumed to be in Earth-centered, Earth-fixed (ECEF) coordinates.

    Returns
    -------
    OrbitPassDirection
        The pass direction of the majority of sampled time points within the orbit.

    Raises
    ------
    AmbiguousOrbitPassDirection
        If the pass direction changes more than once during the duration of the input
        orbit.

    Notes
    -----
    If the orbit contains an equal number of ascending and descending samples, it will
    be considered a descending pass.
    """
    if orbit.size < 1:
        raise ValueError("input orbit must contain at least one state vector")

    # Get the z-component of velocity of each orbit state vector.
    vz = orbit.velocity[:, 2]

    num_sign_changes = count_sign_changes(vz)
    if num_sign_changes > 1:
        raise AmbiguousOrbitPassDirection(
            f"the z-component of the orbit velocity changes sign {num_sign_changes}"
            " times so no clear orbit pass direction can be determined -- consider"
            " cropping the orbit to a narrower interval"
        )

    # Get the number of ascending and descending time points.
    # Orbit state vectors are in ECEF coordinates so a positive z-velocity indicates an
    # ascending pass and a negative z-velocity indicates a descending pass.
    num_asc = np.sum(vz > 0.0)
    num_desc = np.sum(vz < 0.0)

    # Return the orbit pass direction corresponding to a majority of time points within
    # the orbit (tie goes to descending).
    if (num_asc > num_desc):
        return OrbitPassDirection.ASCENDING
    else:
        return OrbitPassDirection.DESCENDING
