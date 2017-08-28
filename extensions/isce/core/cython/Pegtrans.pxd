#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Ellipsoid cimport Ellipsoid
from Peg cimport Peg

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum orbitConvMethod:
        SCH_2_XYZ = 0
        XYZ_2_SCH = 1

cdef extern from "isce/core/Pegtrans.h" namespace "isce::core":
    cdef cppclass Pegtrans:
        vector[vector[double]] mat
        vector[vector[double]] matinv
        vector[double] ov
        double radcur

        Pegtrans(double) except +
        Pegtrans() except +
        Pegtrans(const Pegtrans&) except +
        void radarToXYZ(Ellipsoid&,Peg&)
        void convertSCHtoXYZ(vector[double]&,vector[double]&,orbitConvMethod)
        void convertSCHdotToXYZdot(vector[double]&,vector[double]&,vector[double]&,vector[double]&,
                                   orbitConvMethod)
        void SCHbasis(vector[double]&,vector[vector[double]]&,vector[vector[double]]&)

