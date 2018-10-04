#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

# Get some data types from GDAL
cdef extern from "gdal.h":

    # Get error codes
    ctypedef int CPLErr

    # GDAL RasterBand class
    cdef cppclass GDALRasterBand:
        int GetXSize()
        int GetYSize()
        int GetBand()
        double GetNoDataValue()

    # GDAL Dataset class
    cdef cppclass GDALDataset:
        int GetRasterXSize()
        int GetRasterYSize()
        GDALRasterBand * GetRasterBand(int band)

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

    # Function to open a dataset
    cdef cppclass GDALDatasetH
    GDALDatasetH GDALOpen(const char *, GDALAccess)

# end of file
