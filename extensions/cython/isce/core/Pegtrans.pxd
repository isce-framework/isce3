#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Cartesian cimport cartesian_t, cartmat_t
from Ellipsoid cimport Ellipsoid
from Peg cimport Peg

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum orbitConvMethod:
        SCH_2_XYZ = 0
        XYZ_2_SCH = 1

cdef extern from "isce/core/Pegtrans.h" namespace "isce::core":
    cdef cppclass Pegtrans:
        cartmat_t mat
        cartmat_t matinv
        cartesian_t ov
        double radcur

        Pegtrans(double) except +
        Pegtrans() except +
        Pegtrans(const Pegtrans&) except +
        void radarToXYZ(Ellipsoid&,Peg&)
        void convertSCHtoXYZ(cartesian_t&,cartesian_t&,orbitConvMethod)
        void convertSCHdotToXYZdot(cartesian_t&,cartesian_t&,cartesian_t&,cartesian_t&,
                                   orbitConvMethod)
        void SCHbasis(cartesian_t&,cartmat_t&,cartmat_t&)
