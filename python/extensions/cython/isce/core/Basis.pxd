#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from Cartesian cimport cartesian_t

cdef extern from "isce3/core/Basis.h" namespace "isce3::core":

    # The Basis class
    cdef cppclass Basis:

        # Constructors
        Basis() except +
        Basis(cartesian_t &, cartesian_t &, cartesian_t &) except +
        Basis(cartesian_t & position, cartesian_t & velocity) except +

        # Getters
        cartesian_t x0()
        cartesian_t x1()
        cartesian_t x2()

        # Setters
        void x0(cartesian_t &)
        void x1(cartesian_t &)
        void x2(cartesian_t &)

# end of file
