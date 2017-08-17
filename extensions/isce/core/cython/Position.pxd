#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector

cdef extern from "Position.h" namespace "isceLib":
    cdef cppclass Position:
        vector[double] j
        vector[double] jdot
        vector[double] jddt

        Position() except +
        Position(const Position&) except +
        void lookVec(double,double,vector[double]&)

