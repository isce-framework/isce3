#cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from Cartesian cimport Vec3
from DateTime cimport DateTime
from IH5 cimport IGroup
from StateVector cimport StateVector

cdef extern from "isce/core/Orbit.h" namespace "isce::core":

    cdef enum OrbitInterpMethod:
        Hermite "isce::core::OrbitInterpMethod::Hermite"
        Legendre "isce::core::OrbitInterpMethod::Legendre"

    cdef enum OrbitInterpBorderMode:
        Error "isce::core::OrbitInterpBorderMode::Error"
        Extrapolate "isce::core::OrbitInterpBorderMode::Extrapolate"
        FillNaN "isce::core::OrbitInterpBorderMode::FillNaN"

    cdef cppclass Orbit:
        Orbit()
        Orbit(const vector[StateVector] &) except +
        Orbit(const vector[StateVector] &, const DateTime &) except +
        vector[StateVector] getStateVectors()
        void setStateVectors(const vector[StateVector] &) except +
        const DateTime & referenceEpoch()
        void referenceEpoch(const DateTime &)
        OrbitInterpMethod interpMethod()
        void interpMethod(OrbitInterpMethod)
        double startTime()
        double midTime()
        double endTime()
        DateTime startDateTime()
        DateTime midDateTime()
        DateTime endDateTime()
        double spacing()
        int size()
        double time(int)
        Vec3 position(int)
        Vec3 velocity(int)
        void interpolate(Vec3 *, Vec3 *, double, OrbitInterpBorderMode) except +

    bool operator==(const Orbit &, const Orbit &)
    bool operator!=(const Orbit &, const Orbit &)

cdef extern from "isce/core/Serialization.h" namespace "isce::core":
    void saveOrbitToH5 "saveToH5" (IGroup &, const Orbit &)
    void loadOrbitFromH5 "loadFromH5" (IGroup &, Orbit &)
