#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Cartesian cimport cartesian_t

cdef extern from "isce/core/Position.h" namespace "isce::core":
    cdef cppclass Position:
        vector[double] j
        vector[double] jdot
        vector[double] jddt

        Position() except +
        Position(const Position&) except +
        void lookVec(double,double,cartesian_t&)
