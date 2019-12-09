#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from Cartesian cimport cartesian_t, cartmat_t
from IH5 cimport IGroup

cdef extern from "isce/core/Attitude.h" namespace "isce::core":
    cdef cppclass Attitude:
        cartesian_t ypr(double t)
        cartmat_t rotmat(double t, string)

cdef extern from "isce/core/EulerAngles.h" namespace "isce::core":
    cdef cppclass EulerAngles(Attitude):
        # Getter functions for attitude angles
        const vector[double] yaw()
        const vector[double] pitch()
        const vector[double] roll()
        # Interpolate for all Euler angles at a given time
        void ypr(double t, double & yaw, double & pitch, double & roll)
        # Constructors
        EulerAngles(string) except +
        EulerAngles(const vector[double] & time,
                    const vector[double] & yaw,
                    const vector[double] & pitch,
                    const vector[double] & roll,
                    string) except +
        # Convert to quaternion
        vector[double] toQuaternionElements(double t)

# Wrapper around isce::core serialization defined in <isce/core/Serialization.h
cdef extern from "isce/core/Serialization.h" namespace "isce::core":
    # Load attitude data
    void loadEulerAngles "loadFromH5" (IGroup & group, EulerAngles & euler)
    
    # save attitude data
    void saveEulerAngles "saveToH5" (IGroup & group, EulerAngles & euler)

# end of file
