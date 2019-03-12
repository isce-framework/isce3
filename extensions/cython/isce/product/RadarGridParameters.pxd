#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from DateTime cimport DateTime
from Swath cimport Swath

cdef extern from "isce/product/RadarGridParameters.h" namespace "isce::product":
    cdef cppclass RadarGridParameters:

        # Constructors
        RadarGridParameters() except +
        RadarGridParameters(const Swath & swath,
                            size_t numberAzimuthLooks,
                            size_t numberRangeLooks) except +
        RadarGridParameters(const RadarGridParameters & radargrid) except +

        # Number of azimuth looks
        size_t numberAzimuthLooks()
        
        # Number of range looks
        size_t numberRangeLooks()

        # Reference epoch
        DateTime & refEpoch()

        # Sensing start in seconds
        double sensingStart()

        # Wavelength
        double wavelength()

        # Pulse repetition frequency
        double prf()

        # Starting slant range
        double startingRange()

        # Range pixel spacing
        double rangePixelSpacing()

        # Radar grid length
        size_t length()

        # Radar grid width
        size_t width()

        # Number of radar grid elements
        size_t size()

        # Sensing stop in seconds
        double sensingStop()

        # Sensing mid in seconds
        double sensingMid()

        # Sensing time for a given line
        double sensingTime(size_t)

        # Sensing DateTime for a given line
        DateTime sensingDateTime(size_t)

        # Endling slant range
        double endingRange()

        # Mid slant range
        double midRange()

        # Slant range for a given sample
        double slantRange(size_t)

# end of file
