from dataclasses import astuple, dataclass

import numpy as np


@dataclass
class LLH:
    """
    A point in a geodetic coordinate system.

    Parameters
    ----------
    longitude : float
        The geodetic longitude, in radians.
    latitude : float
        The geodetic latitude, in radians.
    height : float
        The height of the point above the reference ellipsoid, in meters.
    """

    longitude: float
    latitude: float
    height: float

    def to_vec3(self) -> np.ndarray:
        """Convert the LLH object into an array containing [lon, lat, height]."""
        return np.asarray(astuple(self))
