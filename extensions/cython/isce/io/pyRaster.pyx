#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from osgeo import gdal
import h5py
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t
from Raster cimport Raster
from GDAL cimport GDALDataset, GDALAccess, GDALRegister_IH5
from GDAL cimport GDALDataType as GDT

cdef class pyRaster:
    '''
    Python wrapper for isce::core::Raster

    All parameters like dimensions, data types etc must be known at the time of creation.

    Args:
        filename (str): Filename on disk to create or to read
        access (Optional[int]): gdal.GA_ReadOnly or gdal.GA_Update
        dtype (Optional[int]): gdal.GDT_* for creating new raster
        width (Optional[int]): width of new raster to be created
        length (Optional[int]): length of new raster to created
        numBands (Optional[int]): number of bands in new raster to be created
        driver (Optional[int]): GDAL driver to use for creation
    '''

    cdef Raster * c_raster
    cdef bool __owner

    def __cinit__(self, py_filename, int access=0, int dtype=0, int width=0,
                  int length=0, int numBands=0, driver='', collection=[], 
                  dataset=None, h5=None):

        # If a gdal.Dataset is passed in as a keyword argument, intercept that here
        # and create a Raster
        cdef GDALDataset * gdal_dset
        cdef uint64_t swig_pointer
        cdef string dsname
        if h5 is not None:
            assert isinstance(h5, h5py.Dataset), \
                'h5 must be a h5py.Dataset instance.'

            GDALRegister_IH5()
            dsname = pyStringToBytes('IH5:::ID={0}'.format(h5.id.id))
            if (access):
                self.c_raster = new Raster(dsname, GDALAccess.GA_Update)
            else:
                self.c_raster = new Raster(dsname)

            self.__owner = False
            return


        if dataset is not None:
            assert isinstance(dataset, gdal.Dataset), \
                'dataset must be a gdal.Dataset instance.'
            # Get swig address to underlying GDAL dataset
            swig_pointer = <uint64_t> dataset.this
            # Convert to cython GDALDataset
            gdal_dset = <GDALDataset *> swig_pointer
            # Make raster
            self.c_raster = new Raster(gdal_dset)
            self.__owner = False
            return
        
        # Convert the filename to a C++ string representation
        cdef string filename = pyStringToBytes(py_filename)
        cdef string drivername = pyStringToBytes(driver)
        
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

        cdef vector[Raster] rasterlist 
        if collection:
            for inobj in collection:
                rasterlist.push_back( (<pyRaster>inobj).c_raster[0] )

            self.c_raster = new Raster(filename, rasterlist)
            return
        
        # Read-only
        if access == 0:
            self.c_raster = new Raster(filename)

        # New file
        else:
            if drivername.empty():
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

            else:
                if gdtype == GDT.GDT_Unknown:
                    raise ValueError('Cannot create raster with unknown data type')
                else:
                    self.c_raster = new Raster(filename, width, length, numBands,
                                            gdtype, drivername)

        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_raster

    # Get width
    @property
    def width(self):
        '''
        Return width of the raster.

        Returns:
            int : Width of the raster
        '''
        return self.c_raster.width()

    # Get length
    @property
    def length(self):
        '''
        Return length of the raster.

        Returns:
            int : Length of the raster
        '''
        return self.c_raster.length()

    # Get number of bands
    @property
    def numBands(self):
        '''
        Return number of Bands in the raster.

        Returns:
            int : Number of bands in the raster.
        '''
        return self.c_raster.numBands()

    def getDatatype(self, int band=1):
        '''
        Return GDAL data type of Raster.

        Returns:
            int: gdal.GDT_* datatype
        '''
        return self.c_raster.dtype(band)

    #Return if raster can be updated.
    @property
    def isReadOnly(self):
        '''
        Returns flag to indicate if raster is read only.

        Returns:
            bool: Flag indicating if raster is read only.
        '''

        return self.c_raster.access() == 0

    @property
    def EPSG(self):
        '''
        Returns EPSG code associated with raster.

        Returns:
            int: EPSG code
        '''
        return self.c_raster.getEPSG()

    @EPSG.setter
    def EPSG(self, code):
        '''
        Set EPSG code.

        Args:
            code (int): EPSG code

        Returns:
            None
        '''
        self.c_raster.setEPSG(code)

    def addRasterToVRT(self, pyRaster raster):
        '''
        Add Raster to VRT Raster.

        Args:
            raster(pyRaster): Another instance of raster

        Returns:
            None
        '''

        self.c_raster.addRasterToVRT(raster.c_raster[0])

# end of file        
