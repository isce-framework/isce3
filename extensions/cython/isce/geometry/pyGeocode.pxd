#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from Geocode cimport Geocode

cdef class pyGeocodeBase:

    # isce core objects
    cdef Orbit * c_orbit
    cdef Ellipsoid * c_ellipsoid
    cdef LUT2d[double] * c_doppler
    cdef string refepoch_string

    # Processing options
    cdef double threshold
    cdef int numiter
    cdef int linesPerBlock
    cdef double demBlockMargin
    cdef int radarBlockMargin
    cdef dataInterpMethod interpMethod

    # Radar grid parameters
    cdef double azimuthStartTime
    cdef double azimuthTimeInterval
    cdef int radarGridLength
    cdef double startingRange
    cdef double rangeSpacing
    cdef double wavelength
    cdef int radarGridWidth
    cdef int lookSide

    # Geographic grid parameters
    cdef double geoGridStartX
    cdef double geoGridStartY
    cdef double geoGridSpacingX
    cdef double geoGridSpacingY
    cdef int width
    cdef int length
    cdef int epsgcode

cdef class pyGeocodeFloat(pyGeocodeBase):
    pass

cdef class pyGeocodeDouble(pyGeocodeBase):
    pass

cdef class pyGeocodeComplexFloat(pyGeocodeBase):
    pass

# end of file
