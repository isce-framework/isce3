#cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from Ellipsoid cimport Ellipsoid

cdef extern from "isce/core/Attitude.h" namespace "isce::core":

    cdef cppclass Quaternion:
        vector[double] qvec
        Quaternion(vector[double]) except +
        vector[double] ypr()
        vector[double] factoredYPR(vector[double], vector[double], Ellipsoid *)
        vector[vector[double]] rotmat(string)

    cdef cppclass EulerAngles:
        double yaw
        double pitch
        double roll
        EulerAngles(double, double, double, bool) except +
        vector[double] ypr()
        vector[vector[double]] rotmat(string)
        vector[double] toQuaternionElements()

# end of file
