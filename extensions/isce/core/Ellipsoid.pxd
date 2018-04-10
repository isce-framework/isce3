#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Cartesian cimport cartesian_t

cdef extern from "isce/core/Ellipsoid.h" namespace "isce::core":
    cdef cppclass Ellipsoid:
        Ellipsoid() except +
        Ellipsoid(double,double) except +
        Ellipsoid(const Ellipsoid&) except +
        double a()
        double e2()
        void a(double)
        void e2(double) 
        double rEast(double)
        double rNorth(double)
        double rDir(double,double)
        #void latLon(cartesian_t&,cartesian_t&,latLonConvMethod)
        void latLonToXyz(cartesian_t&,cartesian_t&)
        void xyzToLatLon(cartesian_t&,cartesian_t&)
        void getAngs(cartesian_t&,cartesian_t&,cartesian_t&,double&,double&)
        void getTCN_TCvec(cartesian_t&,cartesian_t&,cartesian_t&,cartesian_t&)
        void TCNbasis(cartesian_t&,cartesian_t&,cartesian_t&,cartesian_t&,
                      cartesian_t&)
