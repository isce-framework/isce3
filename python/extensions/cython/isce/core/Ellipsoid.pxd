#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from libcpp cimport bool
from Cartesian cimport cartesian_t

cdef extern from "isce/core/Ellipsoid.h" namespace "isce::core":
    cdef cppclass Ellipsoid:
        Ellipsoid() except +
        Ellipsoid(double,double) except +
        Ellipsoid(const Ellipsoid&) except +
        double a()
        double e2()
        double b()
        void a(double)
        void e2(double) 
        double rEast(double)
        double rNorth(double)
        double rDir(double,double)
        void lonLatToXyz(cartesian_t&,cartesian_t&)
        void xyzToLonLat(cartesian_t&,cartesian_t&)
        void getImagingAnglesAtPlatform(cartesian_t&,cartesian_t&,cartesian_t&,double&,double&)

# end of file
