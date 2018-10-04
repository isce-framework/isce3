#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from DateTime cimport DateTime

cdef extern from "isce/product/ImageMode.h" namespace "isce::product":
   
    # Image mode class
    cdef cppclass ImageMode:

        # Constructors
        ImageMode() except +
        ImageMode(const ImageMode &) except +
        ImageMode(const string &) except +

        # Image dimensions
        size_t length()
        size_t width()

        # Mode type ('aux' or 'primary')
        string modeType()

        # HDF5 path to dataset
        string dataPath(const string &)

        # PRF
        double prf()
        void prf(double)

        # Bandwidth
        double rangeBandwidth()
        void rangeBandwidth(double)

        # Wavelength
        double wavelength()
        void wavelength(double)

        # Starting range
        double startingRange()
        void startingRange(double)

        # Range pixel spacing
        double rangePixelSpacing()
        void rangePixelSpacing(double)

        # Number of azimuth looks
        size_t numberAzimuthLooks()
        void numberAzimuthLooks(size_t)

        # Number of range looks
        size_t numberRangeLooks()
        void numberRangeLooks(size_t)

        # Zero doppler starting azimuth time
        DateTime startAzTime()
        void startAzTime(const DateTime &) 

        # Zero dopple ending azimuth time
        DateTime endAzTime()
        void endAzTime(const DateTime &)


# end of file 
