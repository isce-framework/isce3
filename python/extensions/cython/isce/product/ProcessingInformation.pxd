#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from Matrix cimport valarray
from LUT2d cimport LUT2d

# ProcessingInformation
cdef extern from "isce/product/ProcessingInformation.h" namespace "isce::product":

    cdef cppclass ProcessingInformation:

        # Default constructor
        ProcessingInformation() except +

        # Copy constructor
        ProcessingInformation(const ProcessingInformation &) except +

        # Slant range coordinates
        const valarray[double] & slantRange()
        void slantRange(const valarray[double] &)

        # Zero doppler time coordinates
        const valarray[double] & zeroDopplerTime()
        void zeroDopplerTime(const valarray[double] &)

        # Effective velocity
        const LUT2d[double] & effectiveVelocity()
        void effectiveVelocity(const LUT2d[double] &)

        # Azimuth FM rate
        const LUT2d[double] & azimuthFMRate(char)
        void azimuthFMRate(const LUT2d[double] &, char)

        # Doppler centroid
        const LUT2d[double] & dopplerCentroid(char)
        void dopplerCentroid(const LUT2d[double] &, char)

# end of file
