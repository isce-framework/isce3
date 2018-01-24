#cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from Ellipsoid cimport Ellipsoid

cdef extern from "isce/core/Attitude.h" namespace "isce::core":

    cdef cppclass Attitude:
        vector[double] ypr()
        vector[vector[double]] rotmat(string)
    
    cdef cppclass Quaternion(Attitude):
        vector[double] qvec
        Quaternion(vector[double]) except +
        vector[double] factoredYPR(vector[double], vector[double], Ellipsoid *)

    cdef cppclass EulerAngles(Attitude):
        double yaw
        double pitch
        double roll
        EulerAngles(double, double, double, string, bool) except +
        vector[double] toQuaternionElements()

# end of file
