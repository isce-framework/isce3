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
from Orbit cimport orbitInterpMethod
from Interpolator cimport dataInterpMethod

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

    # Orbit interpolation methods
    orbitInterpMethods = {
        'hermite': orbitInterpMethod.HERMITE_METHOD,
        'sch' :  orbitInterpMethod.SCH_METHOD,
        'legendre': orbitInterpMethod.LEGENDRE_METHOD
    }

    # DEM interpolation methods
    demInterpMethods = {
        'sinc': dataInterpMethod.SINC_METHOD,
        'bilinear': dataInterpMethod.BILINEAR_METHOD,
        'bicubic': dataInterpMethod.BICUBIC_METHOD,
        'nearest': dataInterpMethod.NEAREST_METHOD,
        'akima': dataInterpMethod.AKIMA_METHOD,
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }

    def __cinit__(self, pyProduct product):
        """
        Constructor takes in a product in order to retrieve relevant radar parameters.
        """
        self.c_topo = new Topo(deref(product.c_product))
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_topo

    def topo(self, pyRaster demRaster, outputDir, threshold=0.05, numIterations=25,
             extraIterations=10, orbitMethod='hermite', demMethod='biquintic', epsgOut=4326):
        """
        Run topo.
        
        Args:
            demRaster (pyRaster):               Raster for input DEM.
            outputDir (str):                    String for output directory.
            threshold (Optional[float]):        Threshold for iteration stop for slant range.
            numIterations (Optional[int]):      Max number of normal iterations.
            extraIterations (Optional[int]):    Number of extra refinement iterations.
            orbitMethod (Optional[str]):        Orbit interpolation method
                                                    ('hermite', 'sch', 'legendre')
            demMethod (Optional[int]):          DEM interpolation method
                                                    ('sinc', 'bilinear', 'bicubic', 'nearest',
                                                     'akima', 'biquintic')
            epsgOut (Optional[int]):            EPSG code for output topo layers.

        Return:
            None
        """
        # Set processing options
        self.c_topo.threshold(threshold)
        self.c_topo.numiter(numIterations)
        self.c_topo.extraiter(extraIterations)
        self.c_topo.orbitMethod(self.orbitInterpMethods[orbitMethod])
        self.c_topo.demMethod(self.demInterpMethods[demMethod])
        self.c_topo.epsgOut(epsgOut)

        # Convert output directory to C++ string
        cdef string outdir = pyStringToBytes(outputDir)
            
        # Run topo
        self.c_topo.topo(deref(demRaster.c_raster), outdir)

# end of file
