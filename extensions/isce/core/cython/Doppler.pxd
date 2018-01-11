#cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Orbit cimport Orbit
from Ellipsoid cimport Ellipsoid

cdef extern from "isce/core/Doppler.h" namespace "isce::core":
    cdef cppclass Doppler[Attitude]:
        vector[double] satxyz
        vector[double] satvel
        vector[double] satllh
        Doppler(Orbit *, Attitude *, Ellipsoid *, double) except +
        double centroid(double, double, string, int, int, bool)

# end of file
