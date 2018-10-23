#cython: language_level=3
#
# Author: Joshua Cohen, Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.vector cimport vector
from Matrix cimport valarray, Matrix

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum dataInterpMethod:
        SINC_METHOD = 0
        BILINEAR_METHOD = 1
        BICUBIC_METHOD = 2
        NEAREST_METHOD = 3
        AKIMA_METHOD = 4
        BIQUINTIC_METHOD = 5

cdef extern from "isce/core/Interpolator.h" namespace "isce::core":

    # Base interpolator class
    cdef cppclass Interpolator[U]:
        U interpolate(double x, double y, const Matrix[U] & z)


cdef extern from "isce/core/Interpolator.h" namespace "isce::core":

    # Base interpolator class
    #cdef cppclass Interpolator[U]:
    #    interpolate(double x, double y, const Matrix[U] & z)

    # Bilinear interpolator class
    cdef cppclass BilinearInterpolator[U](Interpolator[U]):
        BilinearInterpolator() except +

    # Bicubic interpolator class
    cdef cppclass BicubicInterpolator[U](Interpolator[U]):
        BicubicInterpolator() except +

    # 2D Spline interpolator class
    cdef cppclass Spline2dInterpolator[U](Interpolator[U]):
        Spline2dInterpolator(size_t order) except +


# end of file
