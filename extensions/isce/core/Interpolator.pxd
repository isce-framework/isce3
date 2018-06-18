#cython: language_level=3
#
# Author: Joshua Cohen, Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.vector cimport vector
from Matrix cimport valarray, Matrix

cdef extern from "isce/core/Interpolator.h" namespace "isce::core":
    cdef cppclass Interpolator:
        Interpolator() except +
        @staticmethod
        U bilinear[U](double, double, const Matrix[U] &)
        @staticmethod
        U bicubic[U](double, double, const Matrix[U] &)
        @staticmethod
        void sinc_coef(double, double, int, double, int, valarray[double] &)
        @staticmethod
        U sinc_eval[U,V](const Matrix[U] &, const Matrix[V] &, int, int, double, int)
        @staticmethod
        U sinc_eval_2d[U,V](const Matrix[U] &, const Matrix[V] &,
                            int, int, double, double, int, int)
        @staticmethod
        float interp_2d_spline[U](int, const Matrix[U] &, double, double)
        @staticmethod
        double quadInterpolate(valarray[double] &, valarray[double] &, double)
        @staticmethod
        double akima(double, double, const Matrix[float] &)

# end of file
