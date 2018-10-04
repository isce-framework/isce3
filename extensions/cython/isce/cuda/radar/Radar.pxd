#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Poly2d cimport Poly2d

cdef extern from "isce/radar/Radar.h" namespace "isce::radar":

    # Radar class
    cdef cppclass Radar:

        # Constructors
        Radar() except +
        Radar(const Poly2d & skew, const Poly2d & content) except +
        Radar(const Radar &) except +

        # Get content Doppler polynomial
        Poly2d contentDoppler()
        # Set content Doppler polynomial
        void contentDoppler(const Poly2d &)

        # Get skew Doppler polynomial
        Poly2d skewDoppler()
        # Set skew Doppler polynomial
        void skewDoppler(const Poly2d &)

# end of file
