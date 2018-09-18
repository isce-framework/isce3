#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Orbit cimport Orbit
from Radar cimport Radar
from Identification cimport Identification

cdef extern from "isce/product/Metadata.h" namespace "isce::product":

    # Metadata class
    cdef cppclass Metadata:

        # Constructors
        Metadata() except +

        # The NOE orbit
        Orbit orbitNOE()
        void orbitNOE(const Orbit &)

        # The POE orbit
        Orbit orbitPOE()
        void orbitPOE(const Orbit &)

        # The radar instrument
        Radar instrument()
        void instrument(const Radar &)

        # The identification
        Identification identification()
        void identification(const Identification &)


# end of file
