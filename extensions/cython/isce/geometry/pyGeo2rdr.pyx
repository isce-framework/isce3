#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string

from Geo2rdr cimport *

cdef class pyGeo2rdr:
    """
    Cython wrapper for isce::geometry::Geo2rdr.

    Args:
        product (pyProduct):                 Configured Product.

    Return:
        None
    """
    # C++ class instances
    cdef Geo2rdr * c_geo2rdr
    cdef bool __owner

    def __cinit__(self, pyProduct product):
        """
        Constructor takes in a product in order to retrieve relevant radar parameters.
        """
        self.c_geo2rdr = new Geo2rdr(deref(product.c_product))
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_geo2rdr

    def geo2rdr(self, pyRaster topoRaster, outputDir, double azshift=0.0, double rgshift=0.0):
        """
        Run geo2rdr.
        
        Args:
            topoRaster (pyRaster):              Raster for topo products.
            outputDir (str):                    String for output directory.
            azshift (Optional[double]):         Constant azimuth offset.
            rgshift (Optional[double]):         Constant range offset.

        Return:
            None
        """
        cdef string outdir = pyStringToBytes(outputDir)
        self.c_geo2rdr.geo2rdr(deref(topoRaster.c_raster), outdir, azshift, rgshift)


    def geo2rdr_temp(self, pyRaster topoRaster, pyRaster rgoffRaster, pyRaster azoffRaster,
                     double azshift=0.0, double rgshift=0.0):

        self.c_geo2rdr.geo2rdr(deref(topoRaster.c_raster), deref(rgoffRaster.c_raster),
                               deref(azoffRaster.c_raster), azshift, rgshift)


# end of file
