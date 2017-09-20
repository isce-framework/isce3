#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector

cdef extern from "Constants.h" namespace "isce::core":
    cdef enum latLonConvMethod:
        LLH_2_XYZ = 0
        XYZ_2_LLH = 1
        XYZ_2_LLH_OLD = 2

cdef extern from "Ellipsoid.h" namespace "isce::core":
    cdef cppclass Ellipsoid:
        double a
        double e2

        Ellipsoid() except +
        Ellipsoid(double,double) except +
        Ellipsoid(const Ellipsoid&) except +
        double rEast(double)
        double rNorth(double)
        double rDir(double,double)
        void latLon(vector[double]&,vector[double]&,latLonConvMethod)
        void getAngs(vector[double]&,vector[double]&,vector[double]&,double&,double&)
        void getTCN_TCvec(vector[double]&,vector[double]&,vector[double]&,vector[double]&)
        void TCNbasis(vector[double]&,vector[double]&,vector[double]&,vector[double]&,
                      vector[double]&)

