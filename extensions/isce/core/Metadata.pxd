#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from DateTime cimport DateTime

cdef extern from "isce/core/Metadata.h" namespace "isce::core":
    cdef cppclass Metadata:
        Metadata() except +
        double radarWavelength
        double prf;
        double rangeFirstSample;
        double slantRangePixelSpacing;
        double pulseDuration;
        double chirpSlope;
        double antennaLength;
        int lookSide;
        DateTime sensingStart;
        double pegHeading
        double pegLatitude
        double pegLongitude;
        int numberRangeLooks;
        int numberAzimuthLooks;
        int width;
        int length;

# end of file
