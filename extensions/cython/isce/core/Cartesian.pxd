#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

cdef extern from "isce/core/Cartesian.h" namespace "isce::core" nogil:

    # Three-element array for representing coordinate vectors
    cdef cppclass cartesian_t "isce::core::cartesian_t":
        cartesian_t() except +
        double& operator[](size_t)

    # Three-by-three matrix
    cdef cppclass cartmat_t "isce::core::cartmat_t":
        cartmat_t() except +
        cartesian_t& operator[](size_t)
