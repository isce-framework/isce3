#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from Matrix cimport valarray

# LUT1d
cdef extern from "isce/core/LUT1d.h" namespace "isce::core":
    cdef cppclass LUT1d[T]:

        # Constructors
        LUT1d() except +
        LUT1d(const valarray[double] &, const valarray[T], bool) except +
        LUT1d(const LUT1d[T] &) except +

        # Setters and getters
        valarray[double] coords()
        void coords(const valarray[double] &)
        valarray[T] values()
        void values(const valarray[T] &)
        bool extrapolate()
        void extrapolate(bool)

        # Evaluation
        T eval(double)

# end of file 
