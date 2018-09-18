#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

cdef extern from "<array>" namespace "std" nogil:

    # Three-element array for representing coordinate vectors
    cdef cppclass cartesian_t "std::array<double, 3>":
        cartesian_t() except +
        double & operator[](size_t)

    # Three-by-three matrix
    cdef cppclass cartmat_t "std::array<std::array<double, 3>, 3>":
        cartmat_t() except +
        cartesian_t & operator[](size_t)

# end of file
