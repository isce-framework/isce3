#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from Cartesian cimport cartesian_t

cdef extern from "isce/core/Basis.h" namespace "isce::core":

    # The Basis class
    cdef cppclass Basis:

        # Constructors
        Basis() except +
        Basis(cartesian_t &, cartesian_t &, cartesian_t &) except +

        # Getters
        cartesian_t x0()
        cartesian_t x1()
        cartesian_t x2()

        # Setters
        void x0(cartesian_t &)
        void x1(cartesian_t &)
        void x2(cartesian_t &)

# end of file
