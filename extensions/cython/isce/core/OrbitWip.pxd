#cython: language_level=3
#
# Author: Joshua Cohen, Tamas Gal
# Copyright 2017-2019
#

from libcpp.vector cimport vector
from libcpp cimport bool
from Cartesian cimport cartesian_t
from DateTime cimport DateTime
from TimeDelta cimport TimeDelta
from IH5 cimport IGroup

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum orbitInterpMethod:
        HERMITE_METHOD = 0
        SCH_METHOD = 1
        LEGENDRE_METHOD = 2

cdef extern from "isce/core/StateVector.h" namespace "isce::core":
    cdef cppclass StateVector:
       DateTime datetime
       cartesian_t position
       cartesian_t velocity


cdef extern from "isce/orbit_wip/Orbit.h" namespace "isce::orbit_wip":
    cdef cppclass OrbitWip:
        OrbitWip() except +
        OrbitWip(DateTime&, TimeDelta&, int)
        StateVector operator[](int)

cdef extern from "isce/core/Serialization.h" namespace "isce::core":
    void loadOrbit "loadFromH5" (IGroup & group, OrbitWip & orbit)
    void saveOrbit "saveToH5" (IGroup & group, OrbitWip & orbit) 

