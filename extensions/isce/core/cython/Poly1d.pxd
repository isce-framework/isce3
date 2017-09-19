#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector

cdef extern from "Poly1d.h" namespace "isce::core":
    cdef cppclass Poly1d:
        int order
        double mean
        double norm
        vector[double] coeffs

        Poly1d() except +
        Poly1d(int,double,double) except +
        Poly1d(const Poly1d&) except +
        void setCoeff(int,double)
        double getCoeff(int)
        double eval(double)
        void printPoly()

