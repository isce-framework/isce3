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
        threshold (Optional[float]):         Threshold for iteration stop for slant range.
        numIterations (Optional[int]):       Max number of normal iterations.
        orbitMethod (Optional[str]):         Orbit interpolation method
                                                ('hermite', 'sch', 'legendre')

    Return:
        None
    """
    # C++ class instances
    cdef Geo2rdr * c_geo2rdr
    cdef bool __owner

    # Orbit interpolation methods
    orbitInterpMethods = {
        'hermite': orbitInterpMethod.HERMITE_METHOD,
        'sch' :  orbitInterpMethod.SCH_METHOD,
        'legendre': orbitInterpMethod.LEGENDRE_METHOD
    }

    def __cinit__(self, pyProduct product, threshold=1.0e-5, numIterations=50,
                  orbitMethod='hermite'):
        """
        Constructor takes in a product in order to retrieve relevant radar parameters.
        """
        # Create C++ geo2rdr pointer
        self.c_geo2rdr = new Geo2rdr(deref(product.c_product))
        self.__owner = True

        # Set processing options
        self.c_geo2rdr.threshold(threshold)
        self.c_geo2rdr.numiter(numIterations)
        self.c_geo2rdr.orbitMethod(self.orbitInterpMethods[orbitMethod])

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
        # Convert output directory to C++ string
        cdef string outdir = pyStringToBytes(outputDir)

        # Run geo2rdr
        self.c_geo2rdr.geo2rdr(deref(topoRaster.c_raster), outdir, azshift, rgshift)

# end of file
