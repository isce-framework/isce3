#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Ellipsoid cimport Ellipsoid
from Peg cimport Peg

cdef extern from "Pegtrans.h" namespace "isceLib":
    cdef cppclass Pegtrans:
        vector[vector[double]] mat
        vector[vector[double]] matinv
        vector[double] ov
        double radcur

        Pegtrans() except +
        Pegtrans(const Pegtrans&) except +
        void radarToXYZ(Ellipsoid&,Peg&)
        void convertSCHtoXYZ(vector[double]&,vector[double]&,int)
        void convertSCHdotToXYZdot(vector[double]&,vector[double]&,vector[double]&,vector[double]&,int)
        void SCHbasis(vector[double]&,vector[vector[double]]&,vector[vector[double]]&)

