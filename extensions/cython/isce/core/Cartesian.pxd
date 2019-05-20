#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

cdef extern from "isce/core/Cartesian.h" namespace "isce::core" nogil:

    # Three-element array for representing coordinate vectors
    cdef cppclass Vec3 "isce::core::Vec3":
        Vec3() except +
        double& operator[](size_t)

    ctypedef Vec3 cartesian_t

    # Three-by-three matrix
    cdef cppclass Mat3 "isce::core::Mat3":
        Mat3() except +
        Vec3& operator[](size_t)

    ctypedef Mat3 cartmat_t
