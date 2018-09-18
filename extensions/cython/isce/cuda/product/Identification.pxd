#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Ellipsoid cimport Ellipsoid
from libcpp.string cimport string

cdef extern from "isce/product/Identification.h" namespace "isce::product":

    # Identification class
    cdef cppclass Identification:

        # Constructors
        Identification() except +
        Identification(const Identification &)

        # Get look direction
        int lookDirection()
        # Set from integer
        void lookDirection(int)
        # Set from string ('right' or 'left')
        void lookDirection(const string &)

        # Ellipsoid
        Ellipsoid ellipsoid()
        void ellipsoid(const Ellipsoid &)


# end of file
