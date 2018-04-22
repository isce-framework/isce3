#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from cython.operator cimport dereference as deref
from libcpp.string cimport string
cimport Serialization
from Metadata cimport Metadata

cdef class pyMetadata:
    cdef Metadata * c_metadata
    cdef bool __owner

    def __cinit__(self):
        self.c_metadata = new Metadata()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_metadata

    @property
    def radarWavelength(self):
        return self.c_metadata.radarWavelength
    @radarWavelength.setter
    def radarWavelength(self, double value):
        self.c_poly2d.radarWavelength = value

    @property
    def prf(self):
        return self.c_metadata.prf
    @prf.setter
    def prf(self, double value):
        self.c_poly2d.prf = value

    @property
    def rangeFirstSample(self):
        return self.c_metadata.rangeFirstSample
    @rangeFirstSample.setter
    def rangeFirstSample(self, double value):
        self.c_poly2d.rangeFirstSample = value

    @property
    def slantRangePixelSpacing(self):
        return self.c_metadata.slantRangePixelSpacing
    @slantRangePixelSpacing.setter
    def slantRangePixelSpacing(self, double value):
        self.c_poly2d.slantRangePixelSpacing = value

    @property
    def pulseDuration(self):
        return self.c_metadata.pulseDuration
    @pulseDuration.setter
    def pulseDuration(self, double value):
        self.c_poly2d.pulseDuration = value

    @property
    def chirpSlope(self):
        return self.c_metadata.chirpSlope
    @chirpSlope.setter
    def chirpSlope(self, double value):
        self.c_poly2d.chirpSlope = value

    @property
    def antennaLength(self):
        return self.c_metadata.antennaLength
    @antennaLength.setter
    def antennaLength(self, double value):
        self.c_poly2d.antennaLength = value

    @property
    def pegHeading(self):
        return self.c_metadata.pegHeading
    @pegHeading.setter
    def pegHeading(self, double value):
        self.c_poly2d.pegHeading = value

    @property
    def pegLatitude(self):
        return self.c_metadata.pegLatitude
    @pegLatitude.setter
    def pegLatitude(self, double value):
        self.c_poly2d.pegLatitude = value

    @property
    def pegLongitude(self):
        return self.c_metadata.pegLongitude
    @pegLongitude.setter
    def pegLongitude(self, double value):
        self.c_poly2d.pegLongitude = value

    @property
    def lookSide(self):
        return self.c_metadata.lookSide
    @lookSide.setter
    def lookSide(self, int value):
        self.c_poly2d.lookSide = value

    @property
    def numberRangeLooks(self):
        return self.c_metadata.numberRangeLooks
    @numberRangeLooks.setter
    def numberRangeLooks(self, int value):
        self.c_poly2d.numberRangeLooks = value
    
    @property
    def numberAzimuthLooks(self):
        return self.c_metadata.numberAzimuthLooks
    @numberAzimuthLooks.setter
    def numberAzimuthLooks(self, int value):
        self.c_poly2d.numberAzimuthLooks = value

    @property
    def width(self):
        return self.c_metadata.width
    @width.setter
    def width(self, int value):
        self.c_poly2d.width = value

    @property
    def length(self):
        return self.c_metadata.length
    @length.setter
    def length(self, int value):
        self.c_poly2d.length = value

    @property
    def sensingStart(self):
        """
        Return string representation.
        """
        return self.c_metadata.sensingStart.isoformat()
    @sensingStart.setter
    def sensingStart(self, pyDateTime dtime):
        self.c_metadata.sensingStart = deref(dtime.c_datetime)

    def archive(self, string metadata):
        Serialization.load_archive[Metadata](metadata, 'RadarMetadata', self.c_metadata)

# end of file 
