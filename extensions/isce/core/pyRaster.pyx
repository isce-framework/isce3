#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string
from Raster cimport Raster
from Raster cimport GDALDataType as GDT

cdef class pyRaster:
    cdef Raster * c_raster
    cdef bool __owner

    def __cinit__(self, string filename, int access=0, int dtype=0, int width=0,
                  int length=0, int numBands=0):

        # Convert datatypes
        cdef GDT gdtype
        if (dtype == GDT.GDT_Unknown):
            gdtype = GDT.GDT_Unknown
        elif (dtype == GDT.GDT_Byte):
            gdtype = GDT.GDT_Byte
        elif (dtype == GDT.GDT_UInt16):
            gdtype = GDT.GDT_UInt16
        elif (dtype == GDT.GDT_Int16):
            gdtype = GDT.GDT_Int16
        elif (dtype == GDT.GDT_UInt32):
            gdtype = GDT.GDT_UInt32
        elif (dtype == GDT.GDT_Int32):
            gdtype = GDT.GDT_Int32
        elif (dtype == GDT.GDT_Float32):
            gdtype = GDT.GDT_Float32
        elif (dtype == GDT.GDT_Float64):
            gdtype = GDT.GDT_Float64
        elif (dtype == GDT.GDT_CInt16):
            gdtype = GDT.GDT_CInt16
        elif (dtype == GDT.GDT_CInt32):
            gdtype = GDT.GDT_CInt32
        elif (dtype == GDT.GDT_CFloat32):
            gdtype = GDT.GDT_CFloat32
        elif (dtype == GDT.GDT_CFloat64):
            gdtype = GDT.GDT_CFloat64

        # Read-only
        if access == 0:
            self.c_raster = new Raster(filename)

        # New file
        else:
            if gdtype == GDT.GDT_Unknown and width != 0 and length != 0:
                self.c_raster = new Raster(filename, width, length)
        
            elif gdtype == GDT.GDT_Unknown and width != 0 and length != 0 and numBands != 0:
                self.c_raster = new Raster(filename, width, length, numBands)

            elif gdtype != GDT.GDT_Unknown and width != 0 and length != 0:
                self.c_raster = new Raster(filename, width, length, gdtype)

            elif gdtype != GDT.GDT_Unknown and width != 0 and length != 0 and numBands != 0:
                self.c_raster = new Raster(filename, width, length, numBands, gdtype)

            else:
                raise NotImplementedError('Unsupported Raster creation')

        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_raster

    # Get width
    @property
    def width(self):
        return self.c_raster.width()

    # Get length
    @property
    def length(self):
        return self.c_raster.length()

    # Get number of bands
    @property
    def numBands(self):
        return self.c_raster.numBands()

# end of file        
