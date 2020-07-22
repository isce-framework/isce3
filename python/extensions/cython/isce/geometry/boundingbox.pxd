#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2020
#

from Shapes cimport Perimeter
from Orbit cimport Orbit
from Projections cimport ProjectionBase
from LUT2d cimport LUT2d
from DEMInterpolator cimport DEMInterpolator
from RadarGridParameters cimport RadarGridParameters

cdef extern from "isce3/geometry/boundingbox.h" namespace "isce3::geometry":
    Perimeter getGeoPerimeter(RadarGridParameters &,
                              Orbit &,
                              ProjectionBase *,
                              LUT2d[double] &,
                              DEMInterpolator &,
                              int, double, int) except +
