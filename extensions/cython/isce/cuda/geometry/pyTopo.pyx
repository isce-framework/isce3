#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from cython.operator cimport dereference as deref

from SerializeGeometry cimport load_archive
from Topo cimport *

cdef class pyTopo:
    """
    Cython wrapper for isce::geometry::Topo.

    Args:
        product (pyProduct):                 Configured Product.

    Return:
        None
    """
    # C++ class instances
    cdef Topo * c_topo
    cdef bool __owner

    def __cinit__(self, pyProduct product):
        """
        Constructor takes in a product in order to retrieve relevant radar parameters.
        """
        self.c_topo = new Topo(deref(product.c_product))
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_topo

    def topo(self, pyRaster demRaster, outputDir):
        """
        Run topo.
        
        Args:
            demRaster (pyRaster):               Raster for input DEM.
            outputDir (str):                    String for output directory.

        Return:
            None
        """
        cdef string outdir = pyStringToBytes(outputDir)
        self.c_topo.topo(deref(demRaster.c_raster), outdir)

# end of file
