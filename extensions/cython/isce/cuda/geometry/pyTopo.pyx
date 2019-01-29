#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from cython.operator cimport dereference as deref
from isceextension cimport pyProduct
from isceextension cimport pyRaster
from cuTopo cimport *

cdef class pyTopo:
    """
    Cython wrapper for isce::geometry::Topo.

    Args:
        product (pyProduct):                 Configured Product.
        threshold (Optional[float]):         Threshold for iteration stop for slant range.
        numIterations (Optional[int]):       Max number of normal iterations.
        extraIterations (Optional[int]):     Number of extra refinement iterations.
        orbitMethod (Optional[str]):         Orbit interpolation method
                                                 ('hermite', 'sch', 'legendre')
        demMethod (Optional[int]):           DEM interpolation method
                                                 ('sinc', 'bilinear', 'bicubic', 'nearest',
                                                  'akima', 'biquintic')
        epsgOut (Optional[int]):             EPSG code for output topo layers.

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

    def __cinit__(self, pyProduct product, threshold=0.05, numIterations=25,
                  extraIterations=10, orbitMethod='hermite', demMethod='biquintic',
                  epsgOut=4326, computeMask=False):
        """
        Constructor takes in a product in order to retrieve relevant radar parameters.
        """
        # Create C++ topo pointer
        self.c_topo = new Topo(deref(product.c_product))
        self.__owner = True

        # Set processing options
        self.c_topo.threshold(threshold)
        self.c_topo.numiter(numIterations)
        self.c_topo.extraiter(extraIterations)
        self.c_topo.orbitMethod(self.orbitInterpMethods[orbitMethod])
        self.c_topo.demMethod(self.demInterpMethods[demMethod])
        self.c_topo.epsgOut(epsgOut)
        self.c_topo.initialized(True)
        self.c_topo.computeMask(computeMask)

    def __dealloc__(self):
        if self.__owner:
            del self.c_topo

    def topo(self, pyRaster demRaster, pyRaster xRaster=None, pyRaster yRaster=None,
             pyRaster heightRaster=None, pyRaster incRaster=None, pyRaster hdgRaster=None,
             pyRaster localIncRaster=None, pyRaster localPsiRaster=None,
             pyRaster simRaster=None, pyRaster maskRaster=None, outputDir=None):
        """
        Run topo.
        
        Args:
            demRaster (pyRaster):               Raster for input DEM.
            xRaster (Optional[str]):            Raster for output X coordinate.
            yRaster (Optional[str]):            Raster for output Y coordinate.
            heightRaster (Optional[str]):       Raster for output height/Z coordinate.
            incRaster (Optional[str]):          Raster for output incidence angle.
            hdgRaster (Optional[str]):          Raster for output heading angle.
            localIncRaster (Optional[str]):     Raster for output local incidence angle.
            localPsiRaster (Optional[str]):     Raster for output local projection angle.
            simRaster (Optional[str]):          Raster for output simulated amplitude image.
            maskRaster (Optional[str]):         Raster for output layover/shadow mask.
            outputDir (Optional[str]):          String for output directory for internal rasters.

        Return:
            None
        """
        cdef string outdir
        
        # Run topo with pre-created rasters if they exist 
        if xRaster is not None and yRaster is not None and heightRaster is not None:

            # Run with mask computation
            if maskRaster is not None:
                self.c_topo.topo(deref(demRaster.c_raster), deref(xRaster.c_raster),
                                 deref(yRaster.c_raster), deref(heightRaster.c_raster),
                                 deref(incRaster.c_raster), deref(hdgRaster.c_raster),
                                 deref(localIncRaster.c_raster), deref(localPsiRaster.c_raster),
                                 deref(simRaster.c_raster), deref(maskRaster.c_raster))

            # Or without mask computation
            else:
                self.c_topo.topo(deref(demRaster.c_raster), deref(xRaster.c_raster),
                                 deref(yRaster.c_raster), deref(heightRaster.c_raster),
                                 deref(incRaster.c_raster), deref(hdgRaster.c_raster),
                                 deref(localIncRaster.c_raster), deref(localPsiRaster.c_raster),
                                 deref(simRaster.c_raster))

        elif outputDir is not None:
            # Convert output directory to C++ string
            outdir = pyStringToBytes(outputDir)
            # Run topo with internal raster creation
            self.c_topo.topo(deref(demRaster.c_raster), outdir)

        else:
            assert False, 'No rasters or output directory specified for topo'

        
# end of file
