#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

cdef extern from "isce/core/Peg.h" namespace "isce::core":
    cdef cppclass Peg:
        double lat
        double lon
        double hdg

        Peg() except +
        Peg(double,double,double) except +
        Peg(const Peg&) except +

