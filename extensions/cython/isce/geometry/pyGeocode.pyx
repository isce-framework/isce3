#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from libcpp.string cimport string
from cython.operator cimport dereference as deref
cimport cython

from Geocode cimport *

cdef class pyGeocodeBase:
    """
    Cython wrapper for isce::geometry::Geocode.

    Args:
        orbit (pyOrbit):                    Orbit instance.
        ellps (pyEllipsoid):                Ellipsoid instance.
        threshold (Optional[float]):        Threshold for underlying geo2rdr function calls.
        numiter (Optional[int]):            Max number of iterations for underlying geo2rdr.
        linesPerBlock (Optional[int]):      Number of lines per input radar block.
        demBlockMargin (Optional[float]):   DEM block margin in degrees.
        radarBlockMargin (Optional[int]):   Radar block margin.
        interpMethod (Optional[str]):       Image interpolation method
                                                ('sinc', 'bilinear', 'bicubic', 'nearest',
                                                 'biquintic')

    Return:
        None
    """
    cdef Orbit * c_orbit
    cdef Ellipsoid * c_ellipsoid
    cdef LUT2d[double] * c_doppler
    cdef DateTime * c_refepoch

    # Processing options
    cdef double threshold
    cdef int numiter
    cdef int linesPerBlock
    cdef double demBlockMargin
    cdef int radarBlockMargin
    cdef dataInterpMethod interpMethod

    # Radar grid parameters
    cdef double azimuthStartTime
    cdef double azimuthTimeInterval
    cdef int radarGridLength
    cdef double startingRange
    cdef double rangeSpacing
    cdef double wavelength
    cdef int radarGridWidth

    # Geographic grid parameters
    cdef double geoGridStartX
    cdef double geoGridStartY
    cdef double geoGridSpacingX
    cdef double geoGridSpacingY
    cdef int width
    cdef int length
    cdef int epsgcode

    # DEM interpolation methods
    demInterpMethods = {
        'sinc': dataInterpMethod.SINC_METHOD,
        'bilinear': dataInterpMethod.BILINEAR_METHOD,
        'bicubic': dataInterpMethod.BICUBIC_METHOD,
        'nearest': dataInterpMethod.NEAREST_METHOD,
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }

    def __cinit__(self,
                  pyOrbit orbit,
                  pyEllipsoid ellps,
                  double threshold=1.0e-8,
                  int numiter=25,
                  int linesPerBlock=1000,
                  double demBlockMargin=0.1,
                  int radarBlockMargin=10,
                  interpMethod='biquintic'):

        # Save pointers to ISCE objects
        self.c_orbit = orbit.c_orbit
        self.c_ellipsoid = ellps.c_ellipsoid

        # Save geocoding properties
        self.threshold = threshold
        self.numiter = numiter
        self.linesPerBlock = linesPerBlock
        self.demBlockMargin = demBlockMargin
        self.radarBlockMargin = radarBlockMargin
        self.interpMethod = self.demInterpMethods[interpMethod]
        
        return
    
    def radarGrid(self,
                  pyLUT2d doppler,
                  pyDateTime refEpoch,
                  double azimuthStartTime,
                  double azimuthTimeInterval,
                  int radarGridLength,
                  double startingRange,
                  double rangeSpacing,
                  double wavelength,
                  int radarGridWidth):
        """
        Save parameters for radar grid bounds and Doppler representation.
        """
        self.c_doppler = doppler.c_lut
        self.c_refepoch = refEpoch.c_datetime
        self.azimuthStartTime = azimuthStartTime
        self.azimuthTimeInterval = azimuthTimeInterval
        self.radarGridLength = radarGridLength
        self.startingRange = startingRange
        self.rangeSpacing = rangeSpacing
        self.wavelength = wavelength
        self.radarGridWidth = radarGridWidth
        return

    def geoGrid(self,
                double geoGridStartX,
                double geoGridStartY,
                double geoGridSpacingX,
                double geoGridSpacingY,
                int width,
                int length,
                int epsgcode):
        """
        Saves parameters for output geographic grid.
        """
        self.geoGridStartX = geoGridStartX
        self.geoGridStartY = geoGridStartY
        self.geoGridSpacingX = geoGridSpacingX
        self.geoGridSpacingY = geoGridSpacingY
        self.width = width
        self.length = length
        self.epsgcode = epsgcode
        return


cdef class pyGeocodeFloat(pyGeocodeBase):

    def geocode(self,
                pyRaster inputRaster,
                pyRaster outputRaster,
                pyRaster demRaster,
                int inputBand=1):
        """
        Run geocoding.
        """
        # Create Geocoding object
        cdef Geocode[float] c_geocode = Geocode[float]()

        # Set properties
        c_geocode.orbit(deref(self.c_orbit))
        c_geocode.ellipsoid(deref(self.c_ellipsoid))
        c_geocode.thresholdGeo2rdr(self.threshold)
        c_geocode.numiterGeo2rdr(self.numiter)
        c_geocode.linesPerBlock(self.linesPerBlock)
        c_geocode.demBlockMargin(self.demBlockMargin)
        c_geocode.radarBlockMargin(self.radarBlockMargin)
        c_geocode.interpolator(self.interpMethod)

        # Set radar grid
        c_geocode.radarGrid(deref(self.c_doppler), deref(self.c_refepoch),
                            self.azimuthStartTime, self.azimuthTimeInterval,
                            self.radarGridLength, self.startingRange, self.rangeSpacing,
                            self.wavelength, self.radarGridWidth)

        # Set geo grid
        c_geocode.geoGrid(self.geoGridStartX, self.geoGridStartY, self.geoGridSpacingX,
                          self.geoGridSpacingY, self.width, self.length, self.epsgcode)

        # Run geocoding
        c_geocode.geocode(deref(inputRaster.c_raster), deref(outputRaster.c_raster),
                          deref(demRaster.c_raster))
        
        return


cdef class pyGeocodeDouble(pyGeocodeBase):

    def geocode(self,
                pyRaster inputRaster,
                pyRaster outputRaster,
                pyRaster demRaster,
                int inputBand=1):
        """
        Run geocoding.
        """
        # Create Geocoding object
        cdef Geocode[double] c_geocode = Geocode[double]()

        # Set properties
        c_geocode.orbit(deref(self.c_orbit))
        c_geocode.ellipsoid(deref(self.c_ellipsoid))
        c_geocode.thresholdGeo2rdr(self.threshold)
        c_geocode.numiterGeo2rdr(self.numiter)
        c_geocode.linesPerBlock(self.linesPerBlock)
        c_geocode.demBlockMargin(self.demBlockMargin)
        c_geocode.radarBlockMargin(self.radarBlockMargin)
        c_geocode.interpolator(self.interpMethod)

        # Set radar grid
        c_geocode.radarGrid(deref(self.c_doppler), deref(self.c_refepoch),
                            self.azimuthStartTime, self.azimuthTimeInterval,
                            self.radarGridLength, self.startingRange, self.rangeSpacing,
                            self.wavelength, self.radarGridWidth)

        # Set geo grid
        c_geocode.geoGrid(self.geoGridStartX, self.geoGridStartY, self.geoGridSpacingX,
                          self.geoGridSpacingY, self.width, self.length, self.epsgcode)

        # Run geocoding
        c_geocode.geocode(deref(inputRaster.c_raster), deref(outputRaster.c_raster),
                          deref(demRaster.c_raster))
        
        return


# end of file
