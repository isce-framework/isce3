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
        radarGrid (pyRadarGridParameters)    radar grid parameters object .
        orbit     (pyOrbit)                  orbit object
        ellipsoid (pyEllipsoid)              ellipsoid object
        doppler   (pyLUT2d)                  doppler in form of LUT2d object
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

    def __cinit__(self, 
                  pyRadarGridParameters radarGrid,
                  pyOrbit orbit,
                  pyEllipsoid ellipsoid,
                  pyLUT2d doppler=None,
                  double threshold=1.0e-5,
                  int numIterations=50,
                  orbitMethod='hermite'):
        """
        Constructor.
        """
        # Create C++ geo2rdr pointer

        if doppler is None:
            #geometry computation for zero Doppler
            self.c_geo2rdr = new Geo2rdr(deref(radarGrid.c_radargrid), 
                                         deref(orbit.c_orbit),
                                         deref(ellipsoid.c_ellipsoid))
        else:
            self.c_geo2rdr = new Geo2rdr(deref(radarGrid.c_radargrid),
                                         deref(orbit.c_orbit),
                                         deref(ellipsoid.c_ellipsoid),
                                         deref(doppler.c_lut))

        self.__owner = True

        # Set processing options
        self.c_geo2rdr.threshold(threshold)
        self.c_geo2rdr.numiter(numIterations)
        self.c_geo2rdr.orbitMethod(self.orbitInterpMethods[orbitMethod])

    def __dealloc__(self):
        if self.__owner:
            del self.c_geo2rdr

    def geo2rdr(self, pyRaster topoRaster, pyRaster rgoffRaster=None, pyRaster azoffRaster=None,
                outputDir=None, double azshift=0.0, double rgshift=0.0):
        """
        Run geo2rdr.
        
        Args:
            topoRaster (pyRaster):              Raster for input topo products.
            rgoffRaster (Optional[pyRaster]):   Raster for output range offset.
            azoffRaster (Optional[pyRaster]):   Raster for output azimuth offset.
            outputDir (Optional[str]):          String for output directory.
            azshift (Optional[double]):         Constant azimuth offset.
            rgshift (Optional[double]):         Constant range offset. 

        Return:
            None
        """
        cdef string outdir

        if rgoffRaster is not None and azoffRaster is not None:
            # Run geo2rdr directly
            self.c_geo2rdr.geo2rdr(deref(topoRaster.c_raster), deref(rgoffRaster.c_raster),
                                   deref(azoffRaster.c_raster), azshift, rgshift)

        elif outputDir is not None:
            # Convert output directory to C++ string
            outdir = pyStringToBytes(outputDir)
            # Run geo2rdr
            self.c_geo2rdr.geo2rdr(deref(topoRaster.c_raster), outdir, azshift, rgshift)

        else:
            assert False, 'No offset rasters or output directory provided'


# end of file
