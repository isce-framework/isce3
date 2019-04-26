#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from Matrix cimport valarray, Matrix
from Interpolator cimport dataInterpMethod

# LUT2d
cdef extern from "isce/core/LUT2d.h" namespace "isce::core":
    cdef cppclass LUT2d[T]:

        # Constructors
        LUT2d() except +
        LUT2d(double xstart, double ystart, double dx, double dy,
              const Matrix[T] & data, dataInterpMethod method) except +
        LUT2d(const valarray[double] & xcoord,
              const valarray[double] & ycoord,
              const Matrix[T] & data,
              dataInterpMethod method) except +
        LUT2d(const LUT2d[T] &) except +
        
        # Evaluation
        T eval(double x, double y)

# end of file 
