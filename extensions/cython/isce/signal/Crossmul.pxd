#cython: language_level=3
#
# Author: Bryan Riel, Heresh Fattahi
# Copyright 2017-2019
#

from libcpp cimport bool
from LUT1d cimport LUT1d
from Raster cimport Raster

cdef extern from "isce/signal/Crossmul.h" namespace "isce::signal":

    # Class definition
    cdef cppclass Crossmul:

        # Constructor
        Crossmul() except +

        # Setters
        void prf(double)
        void rangeBandwidth(double)
        void rangePixelSpacing(double)
        void wavelength(double)
        void commonAzimuthBandwidth(double)
        void beta(double)
        void rangeLooks(int)
        void azimuthLooks(int)
        void doCommonAzimuthbandFiltering(bool)
        void doCommonRangebandFiltering(bool)

        # Set Doppler profiles from LUT1d objects
        void doppler(LUT1d[double] refDoppler, LUT1d[double] secDoppler)

        # Run default crossmul without range commonband filtering and without 
        # output coherence
        void crossmul(Raster &, Raster &, Raster &)

        # Run default crossmul without range commonband filtering
        void crossmul(Raster &, Raster &, Raster &, Raster &)

        # Run crossmul with offsets to do range commonband filter
        void crossmul(Raster &, Raster &, Raster &, Raster &, Raster &)

# end of file
