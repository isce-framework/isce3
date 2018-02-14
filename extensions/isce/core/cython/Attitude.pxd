#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from Ellipsoid cimport Ellipsoid

cdef extern from "isce/core/Attitude.h" namespace "isce::core":

    cdef cppclass Attitude:
        vector[double] ypr()
        vector[vector[double]] rotmat(string)
    
    cdef cppclass Quaternion(Attitude):
        # Get copy of quaternion elements
        vector[double] getQvec()
        # Set quaternion elements
        void setQvecElement(double, int)
        void setQvec(vector[double])
        # Constructor
        Quaternion(vector[double]) except +
        # Convert quaternion to yaw, pitch, and roll angles
        vector[double] factoredYPR(vector[double], vector[double], Ellipsoid *)

    cdef cppclass EulerAngles(Attitude):
        # Getter functions for attitude angles
        double getYaw()
        double getPitch()
        double getRoll()
        # Setter functions for attitude angles
        void setYaw(double)
        void setPitch(double)
        void setRoll(double)
        # Constructor 
        EulerAngles(double, double, double, string) except +
        # Convert to quaternion
        vector[double] toQuaternionElements()

# end of file
