#-*- coding: utf-8 -*-

# Bryan Riel
# Copyright 2019-

import numpy as np

# The extensions
import isce3.extensions.isceextension as isceextension
from isceextension import pyGeocodeFloat, pyGeocodeDouble, pyGeocodeComplexFloat


def Geocode(orbit=None, ellipsoid=None, inputRaster=None):
    """
    Wrapper for pyGeocode with type specification.
    """
    dtype = inputRaster.getDatatype()

    if isinstance(dtype, int):
        if dtype == 6:
            return pyGeocodeFloat(orbit, ellipsoid)
        elif dtype == 7:
            return pyGeocodeDouble(orbit, ellipsoid)
        elif dtype == 10:
            return pyGeocodeComplexFloat(orbit, ellipsoid)
        else:
            raise NotImplementedError('Geocode data type not yet implemented')
    elif isinstance(dtype, (float, np.float64)):
        return pyGeocodeDouble(orbit, ellipsoid)
    elif isinstance(dtype, np.float32):
        return pyGeocodeFloat(orbit, ellipsoid)
    elif isinstance(dtype, np.complex64):
        return pyGeocodeComplexFloat(orbit, ellipsoid)
    else:
        raise NotImplementedError('Geocode data type not yet implemented')


