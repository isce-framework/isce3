#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from GDAL cimport *

from Error cimport raisePyError

cdef extern from "isce3/io/Raster.h" namespace "isce3::io":

    # Raster class
    cdef cppclass Raster:

        # Constructors
        Raster(const string &) except +raisePyError
        Raster(const string &, GDALAccess) except +
        Raster(const string &, size_t, size_t, size_t, GDALDataType, const string &) except +
        Raster(const string &, size_t, size_t, size_t, GDALDataType) except +
        Raster(const string &, size_t, size_t, size_t) except +
        Raster(const string &, size_t, size_t, GDALDataType) except +
        Raster(const string &, size_t, size_t) except +
        Raster(const string &, const Raster &) except +
        Raster(const Raster &) except +
        Raster(const string &, const vector[Raster] &)
        Raster(GDALDataset *, bool) except +

        # Getters
        Raster & operator=(const Raster &)
        size_t length()
        size_t width()
        size_t numBands()
        GDALAccess access()
        GDALDataType dtype(const size_t)
        bool match(const Raster &)
        void open(string &, GDALAccess)
        int getEPSG()
        void getGeoTransform(double *)
        GDALDataset* dataset()

        # Setters
        void addRasterToVRT(const Raster &) except +raisePyError
        void setEPSG(int)

# end of file
