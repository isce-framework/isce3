#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from LUT1d cimport LUT1d

cdef extern from "isce/radar/Radar.h" namespace "isce::radar":

    # Radar class
    cdef cppclass Radar:

        # Constructors
        Radar() except +
        Radar(const LUT1d[double] & skew, const LUT1d[double] & content) except +
        Radar(const Radar &) except +

        # Get content Doppler LUT
        LUT1d[double] contentDoppler()
        # Set content Doppler LUT
        void contentDoppler(const LUT1d[double] &)

        # Get skew Doppler LUT
        LUT1d[double] skewDoppler()
        # Set skew Doppler LUT
        void skewDoppler(const LUT1d[double] &)

# end of file
