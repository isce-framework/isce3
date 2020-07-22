#cython: language_level=3

from libcpp cimport bool

from Cartesian cimport Vec3
from DateTime cimport DateTime

cdef extern from "isce3/core/StateVector.h" namespace "isce3::core":

    cdef cppclass StateVector:
        StateVector()
        StateVector(const DateTime &, const Vec3 &, const Vec3 &)
        DateTime datetime
        Vec3 position
        Vec3 velocity

    bool operator==(const StateVector &, const StateVector &)
    bool operator!=(const StateVector &, const StateVector &)
