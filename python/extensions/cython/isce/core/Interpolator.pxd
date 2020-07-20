#cython: language_level=3
#
# Author: Joshua Cohen, Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.vector cimport vector
from Matrix cimport valarray, Matrix

cdef extern from "isce3/core/Constants.h" namespace "isce::core":
    cdef enum dataInterpMethod:
        SINC_METHOD
        BILINEAR_METHOD
        BICUBIC_METHOD
        NEAREST_METHOD
        BIQUINTIC_METHOD 

cdef extern from "isce3/core/Interpolator.h" namespace "isce::core":

    # Base interpolator class
    cdef cppclass Interpolator[U]:
        U interpolate(double x, double y, const Matrix[U] & z)


cdef extern from "isce3/core/Interpolator.h" namespace "isce::core":

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
