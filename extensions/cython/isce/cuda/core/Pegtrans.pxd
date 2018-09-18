#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Cartesian cimport cartesian_t, cartmat_t
from Ellipsoid cimport Ellipsoid
from Peg cimport Peg

cdef extern from "isce/core/Pegtrans.h" namespace "isce::core":
    cdef cppclass Pegtrans:
        cartmat_t mat
        cartmat_t matinv
        cartesian_t ov
        double radcur

        Pegtrans(double) except +
        Pegtrans() except +
        Pegtrans(const Pegtrans &) except +
        void radarToXYZ(Ellipsoid &, Peg &)

        void convertXYZtoSCH(const cartesian_t & xyzv, cartesian_t & schv)
        void convertSCHtoXYZ(const cartesian_t & schv, cartesian_t & xyzv)
        void convertXYZdotToSCHdot(const cartesian_t & sch, const cartesian_t & xyzdot,
                                   cartesian_t & schdot)
        void convertSCHdotToXYZdot(const cartesian_t & sch, const cartesian_t & schdot,
                                   cartesian_t & xyzdot)

        void SCHbasis(cartesian_t&,cartmat_t&,cartmat_t&)
