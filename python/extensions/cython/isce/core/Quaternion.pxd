#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from Cartesian cimport cartesian_t, cartmat_t
from Ellipsoid cimport Ellipsoid
from IH5 cimport IGroup

cdef extern from "isce/core/Attitude.h" namespace "isce::core":
    cdef cppclass Attitude:
        cartesian_t ypr()
        cartmat_t rotmat(string)

cdef extern from "isce/core/Quaternion.h" namespace "isce::core": 
    cdef cppclass Quaternion(Attitude):
        # Get copy of quaternion elements
        vector[double] & qvec()
        # Get yaw, pitch, and roll representation
        void ypr(double t, double & yaw, double & pitch, double & roll)
        # Get rotation matrix
        cartmat_t rotmat(double t, const string dummy)
        # Constructor
        Quaternion() except +
        Quaternion(vector[double], vector[double]) except +
        # Convert quaternion to yaw, pitch, and roll angles
        cartesian_t factoredYPR(double, cartesian_t, cartesian_t, Ellipsoid *)

cdef extern from "isce/core/Serialization.h" namespace "isce::core":
    void saveQuaternionToH5 "saveToH5" (IGroup &, const Quaternion &)
    void loadQuaternionFromH5 "loadFromH5" (IGroup &, Quaternion &)
