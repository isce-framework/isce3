#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from libcpp cimport bool
from Cartesian cimport cartesian_t
from DateTime cimport DateTime

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum orbitInterpMethod:
        HERMITE_METHOD = 0
        SCH_METHOD = 1
        LEGENDRE_METHOD = 2

cdef extern from "isce/core/Orbit.h" namespace "isce::core":
    cdef cppclass Orbit:
        int nVectors
        vector[double] position
        vector[double] velocity
        vector[double] UTCtime
        DateTime refEpoch

        Orbit() except +
        Orbit(int) except +
        Orbit(const Orbit&) except +
        void getStateVector(int,double&,cartesian_t&,cartesian_t&)
        void setStateVector(int,double,cartesian_t&,cartesian_t&)
        void addStateVector(double,cartesian_t&,cartesian_t&)
        int interpolate(double,cartesian_t&,cartesian_t&,orbitInterpMethod)
        int interpolateWGS84Orbit(double,cartesian_t&,cartesian_t&)
        int interpolateLegendreOrbit(double,cartesian_t&,cartesian_t&)
        int interpolateSCHOrbit(double,cartesian_t&,cartesian_t&)
        int computeAcceleration(double,cartesian_t&)
        void updateUTCTimes(const DateTime &)
        void printOrbit()
        void loadFromHDR(const char*)
        void dumpToHDR(const char*)

