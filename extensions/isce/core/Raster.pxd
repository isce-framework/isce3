#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string

# Get some values from GDAL
cdef extern from "gdal.h":

    # Get access codes
    ctypedef enum GDALAccess:
        GA_ReadOnly = 0
        GA_Update = 1
    
    # Get datatype codes
    ctypedef enum GDALDataType:
        GDT_Unknown = 0
        GDT_Byte = 1
        GDT_UInt16 = 2
        GDT_Int16 = 3
        GDT_UInt32 = 4
        GDT_Int32 = 5
        GDT_Float32 = 6
        GDT_Float64 = 7
        GDT_CInt16 = 8
        GDT_CInt32 = 9
        GDT_CFloat32 = 10
        GDT_CFloat64 = 11
        GDT_TypeCount = 12

cdef extern from "isce/core/Raster.h" namespace "isce::core":
   
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

        # Getters
        Raster & operator=(const Raster &)
        size_t length()
        size_t width()
        size_t numBands()
        GDALAccess access()
        GDALDataType dtype(const size_t)
        bool match(const Raster &)
        void open(string &, GDALAccess)
        void addRasterToVRT(const Raster &)

# end of file 
