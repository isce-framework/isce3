#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from Matrix cimport valarray

cdef extern from "isce/product/Swath.h" namespace "isce::product":
    cdef cppclass Swath:

        # Constructors
        Swath() except +

        # Slant range coordinates
        const valarray[double] & slantRange()
        void slantRange(const valarray[double] &)

        # Zero doppler time coordinates
        const valarray[double] & zeroDopplerTime()
        void zeroDopplerTime(const valarray[double] &)

        # Get the number of samples
        size_t samples() const
        # Get the number of lines
        size_t lines() const

        # Range pixel spacing
        double rangePixelSpacing() const

        # Acquired center frequency
        double acquiredCenterFrequency() const
        void acquiredCenterFrequency(double)

        # Processed center frequency
        double processedCenterFrequency() const
        void processedCenterFrequency(double)

        # Processed wavelength
        double processedWavelength() const

        # Acquired range bandwidth
        double acquiredRangeBandwidth() const
        void acquiredRangeBandwidth(double)

        # Processed range bandwidth
        double processedRangeBandwidth() const
        void processedRangeBandwidth(double)

        # Nominal acquisition PRF
        double nominalAcquisitionPRF() const
        void nominalAcquisitionPRF(double)

        # Scene center along track spacing
        double sceneCenterAlongTrackSpacing() const
        void sceneCenterAlongTrackSpacing(double)

        # Scene center ground range spacing
        double sceneCenterGroundRangeSpacing() const
        void sceneCenterGroundRangeSpacing(double)

        # Processed bandwidth
        double processedAzimuthBandwidth() const
        void processedAzimuthBandwidth(double)
       
# end of file 
