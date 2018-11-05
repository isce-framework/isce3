#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from GDAL cimport *

cdef extern from "isce/io/Raster.h" namespace "isce::io":
   
    # Raster class
    cdef cppclass Raster:

        # Constructors
        Raster(const string &) except +
        Raster(const string &, GDALAccess) except +
        Raster(const string &, size_t, size_t, size_t, GDALDataType, const string &) except +
        Raster(const string &, size_t, size_t, size_t, GDALDataType) except +
        Raster(const string &, size_t, size_t, size_t) except +
        Raster(const string &, size_t, size_t, GDALDataType) except +
        Raster(const string &, size_t, size_t) except +
        Raster(const string &, const Raster &) except +
        Raster(const Raster &) except +
        Raster(const string &, const vector[Raster] &);
        Raster(GDALDataset *) except +

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
        GDALDataset* dataset()

        # Setters
        void addRasterToVRT(const Raster &)
        void setEPSG(int)

# end of file 
