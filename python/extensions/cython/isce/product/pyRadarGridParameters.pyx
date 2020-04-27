#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from libcpp.string cimport string
from DateTime cimport DateTime
from RadarGridParameters cimport RadarGridParameters
from LookSide cimport LookSide, to_string, parseLookSide

cdef class pyRadarGridParameters:
    """
    Cython wrapper for isce::product::RadarGridParameters.

    Order of parsing keywords is always Swath -> C++ Constructor -> Default constructor
    """
    cdef RadarGridParameters * c_radargrid
    cdef bool __owner

    def __cinit__(self,
                  pySwath swath=None,
                  side = None):
        cdef LookSide _side
        if swath is not None:
            if side is None:
                raise AttributeError("Invalid look direction.")
            _side = pyParseLookSide(side)
            self.c_radargrid = new RadarGridParameters(
                deref(swath.c_swath), _side)
            self.__owner = True
        else:
            self.c_radargrid = new RadarGridParameters()
            self.__owner = True
        return

    def __dealloc__(self):
        if self.__owner:
            del self.c_radargrid

    @staticmethod
    cdef cbind(RadarGridParameters radarGrid):
        """
        Creates a new pyRadarGridParameters instance from a C++ RadarGridParameters instance.
        """
        new_grid = pyRadarGridParameters()
        del new_grid.c_radargrid
        new_grid.c_radargrid = new RadarGridParameters(radarGrid)
        new_grid.__owner = True
        return new_grid

    @property
    def lookSide(self):
        """
        Returns look side ("left" or "right")
        """

        cdef string s = to_string(self.c_radargrid.lookSide())
        return s.decode('UTF-8')

    @lookSide.setter
    def lookSide(self, side):
        """
        Set look side from string "right" or "left"
        """
        self.c_radargrid.lookSide(parseLookSide(pyStringToBytes(side)))

    @property
    def referenceEpoch(self):
        """
        Return reference epoch used to represent azimuth time
        """
        cdef DateTime date = self.c_radargrid.refEpoch()
        return pyDateTime.cbind(date)

    @referenceEpoch.setter
    def referenceEpoch(self, pyDateTime tinp):
        self.c_radargrid.refEpoch(deref(tinp.c_datetime))

    @property
    def sensingStart(self):
        """
        Returns azimuth time corresponding to first line
        """
        return self.c_radargrid.sensingStart()

    @sensingStart.setter
    def sensingStart(self, double val):
        self.c_radargrid.sensingStart(val)

    @property
    def wavelength(self):
        """
        Returns wavelength
        """
        return self.c_radargrid.wavelength()

    @wavelength.setter
    def wavelength(self, double val):
        self.c_radargrid.wavelength(val)

    @property
    def prf(self):
        """
        Returns azimuth time sampling frequency
        """
        return self.c_radargrid.prf()

    @prf.setter
    def prf(self, double val):
        self.c_radargrid.prf(val)

    @property
    def startingRange(self):
        """
        Returns slant range to first pixel in meters
        """
        return self.c_radargrid.startingRange()

    @startingRange.setter
    def startingRange(self, double val):
        self.c_radargrid.startingRange(val)

    @property
    def rangePixelSpacing(self):
        """
        Returns slant range pixel spacing in meters
        """
        return self.c_radargrid.rangePixelSpacing()

    @rangePixelSpacing.setter
    def rangePixelSpacing(self, double val):
        self.c_radargrid.rangePixelSpacing(val)

    @property
    def azimuthTimeInterval(self):
        return self.c_radargrid.azimuthTimeInterval()

    @property
    def length(self):
        """
        Returns number of lines in grid
        """
        return self.c_radargrid.length()

    @length.setter
    def length(self, size_t val):
        self.c_radargrid.length(val)

    @property
    def width(self):
        """
        Returns number of pixels in grid
        """
        return self.c_radargrid.width()

    @width.setter
    def width(self, size_t val):
        self.c_radargrid.width(val)

    @property
    def size(self):
        return self.c_radargrid.size()

    @property
    def sensingStop(self):
        return self.c_radargrid.sensingStop()

    @property
    def sensingMid(self):
        return self.c_radargrid.sensingMid()

    def sensingTime(self, double line):
        return self.c_radargrid.sensingTime(line)

    def sensingDateTime(self, double line):
        cdef DateTime date = self.c_radargrid.sensingDateTime(line)
        return pyDateTime.cbind(date)

    @property
    def endingRange(self):
        return self.c_radargrid.endingRange()

    @property
    def midRange(self):
        return self.c_radargrid.midRange()

    def slantRange(self, double column):
        """
        Returns slant range in meters at given column number
        """
        return self.c_radargrid.slantRange(column)

    def multilook(self, int az_looks, int rg_looks):
        """
        Returns a new grid corresponding to the multilooking parameters
        """
        cdef RadarGridParameters radar_grid = self.c_radargrid.multilook(az_looks, rg_looks)
        return pyRadarGridParameters.cbind(radar_grid)

    def offsetAndResize(self, double yoff, double xoff, size_t ysize, 
            size_t xsize):
        cdef RadarGridParameters radar_grid = self.c_radargrid.offsetAndResize(
            yoff, xoff, ysize, xsize)
        return pyRadarGridParameters.cbind(radar_grid)

    def perimeter(*args, **kwds):
        """
        This is a pass through for isce3.geometry.getPerimeter. Returns parameter as geoJson.
        """
        return getGeoPerimeter(*args, **kwds)

# end of file 
