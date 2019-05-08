#cython: language_level=3
#
# Author: Bryan Riel, Heresh Fattahi, Liang Yu
# Copyright 2017-2019
#

from libcpp cimport bool

from isceextension cimport LUT1d
from isceextension cimport Raster

cdef extern from "isce/cuda/signal/gpuCrossMul.h" namespace "isce::cuda::signal":

    # Class definition
    cdef cppclass gpuCrossmul:

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
        void doCommonAzimuthBandFiltering(bool)
        void doCommonRangeBandFiltering(bool)

        # Set Doppler profiles from LUT1d objects
        void doppler(LUT1d[double] refDoppler, LUT1d[double] secDoppler)

        # Run default crossmul without range commonband filtering
        void crossmul(Raster &, Raster &, Raster &, Raster &)

        # Run crossmul with offsets to do range commonband filter
        void crossmul(Raster &, Raster &, Raster &, Raster &, Raster &)

# end of file
