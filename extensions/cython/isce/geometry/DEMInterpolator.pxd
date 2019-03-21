#cython: language_level=3
#
# Author: Tamas Gal
# Copyright 2019
#


# DEMInterpolator
cdef extern from "isce/geometry/DEMInterpolator.h" namespace "isce::geometry":
    cdef cppclass DEMInterpolator:

        # Constructor
        DEMInterpolator() except +
        DEMInterpolator(double height) except +