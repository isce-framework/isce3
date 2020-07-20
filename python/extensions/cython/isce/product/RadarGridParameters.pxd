#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from DateTime cimport DateTime
from Swath cimport Swath
from LookSide cimport LookSide

cdef extern from "isce3/product/RadarGridParameters.h" namespace "isce::product":
    cdef cppclass RadarGridParameters:

        # Constructors
        RadarGridParameters() except +
        RadarGridParameters(const Swath & swath,
                            LookSide side) except +
        RadarGridParameters(const RadarGridParameters & radargrid) except +
        RadarGridParameters(double sensingStart,
                            double wavelength,
                            double prf,
                            double startingRange,
                            double rangePixelSpacing,
                            LookSide lookSide,
                            size_t length,
                            size_t width,
                            DateTime refEpoch) except +

        # Look side
        LookSide lookSide()
        void lookSide(LookSide side)

        # Reference epoch
        DateTime & refEpoch()
        void refEpoch(DateTime)

        # Sensing start in seconds
        double sensingStart()
        void sensingStart(double)

        # Wavelength
        double wavelength()
        void wavelength(double)

        # Pulse repetition frequency
        double prf()
        void prf(double)

        # Starting slant range
        double startingRange()
        void startingRange(double)

        # Range pixel spacing
        double rangePixelSpacing()
        void rangePixelSpacing(double)

        # Azimuth time interval
        double azimuthTimeInterval()

        # Radar grid length
        size_t length()
        void length(size_t)

        # Radar grid width
        size_t width()
        void width(size_t)

        # Number of radar grid elements
        size_t size()

        # Sensing stop in seconds
        double sensingStop()

        # Sensing mid in seconds
        double sensingMid()

        # Sensing time for a given line
        double sensingTime(double)

        # Sensing DateTime for a given line
        DateTime sensingDateTime(double)

        # Endling slant range
        double endingRange()

        # Mid slant range
        double midRange()

        # Slant range for a given sample
        double slantRange(double)

        RadarGridParameters multilook(size_t, size_t)

        RadarGridParameters offsetAndResize(double, double, size_t, size_t)

# end of file
