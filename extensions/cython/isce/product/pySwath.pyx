#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from libcpp.string cimport string
from Swath cimport Swath, loadSwath

cdef class pySwath:
    """
    Cython wrapper for isce::product::Swath.

    Args:
        None

    Return:
        None
    """
    # C++ class
    cdef Swath * c_swath
    cdef bool __owner

    def __cinit__(self):
        """
        Constructor instantiates a C++ object and saves to python.
        """
        # Create the C++ Swath class
        self.c_swath = new Swath()
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_swath

    @staticmethod
    def bind(pySwath swath):
        """
        Creates a new pySwath instance with C++ Swath attribute shallow copied from
        another C++ Swath attribute contained in a separate instance.

        Args:
            swath (pySwath): External pySwath instance to get C++ Swath from.

        Returns:
            new_swath (pySwath): New pySwath instance with a shallow copy of C++ Swath.
        """
        new_swath = pySwath()
        del new_swath.c_swath
        new_swath.c_swath = swath.c_swath
        new_swath.__owner = False
        return new_swath

    @property
    def samples(self):
        """
        Get number of samples in a radar swath.
        """
        cdef int n = self.c_swath.samples()
        return n

    @property
    def lines(self):
        """
        Get number of lines in a radar swath
        """
        cdef int n = self.c_swath.lines()
        return n

    @property
    def rangePixelSpacing(self):
        """
        Get slant range pixel spacing.
        """
        cdef double r = self.c_swath.rangePixelSpacing()
        return r

    @property
    def acquiredCenterFrequency(self):
        """
        Get acquired center frequency.
        """
        cdef double d = self.c_swath.acquiredCenterFrequency()
        return d

    @property
    def processedCenterFrequency(self):
        """
        Get processed center frequency.
        """
        cdef double d = self.c_swath.processedCenterFrequency()
        return d

    @property
    def processedWavelength(self):
        """
        Get processed wavelength.
        """
        cdef double d = self.c_swath.processedWavelength()
        return d

    @property
    def acquiredRangeBandwidth(self):
        """
        Get acquired range bandwidth.
        """
        cdef double d = self.c_swath.acquiredRangeBandwidth()
        return d

    @property
    def processedRangeBandwidth(self):
        """
        Get processed range bandwidth.
        """
        cdef double d = self.c_swath.processedRangeBandwidth()
        return d

    @property
    def sceneCenterAlongTrackSpacing(self):
        """
        Get scene center along-track spacing.
        """
        cdef double d = self.c_swath.sceneCenterAlongTrackSpacing()
        return d

    @property
    def sceneCenterGroundRangeSpacing(self):
        """
        Get scene center ground range spacing.
        """
        cdef double d = self.c_swath.sceneCenterGroundRangeSpacing()
        return d

    @property
    def processedAzimuthBandwidth(self):
        """
        Get processed azimuth bandwidth.
        """
        cdef double d = self.c_swath.processedAzimuthBandwidth()
        return d

    @staticmethod
    def loadFromH5(h5Group, freq):
        '''
        Load Swath from an HDF5 group

        Args:
            h5Group (h5py group): HDF5 group with swath

        Returns:
            pySwath object
        '''

        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        cdef string freq_str = pyStringToBytes(freq)
        swathObj = pySwath()
        loadSwath(c_igroup, deref(swathObj.c_swath), freq_str[0])
    
    def getRadarGridParameters(self, 
                            numberAzimuthLooks=1,
                            numberRangeLooks=1):
        cdef RadarGridParameters radarGrid = RadarGridParameters(
            deref(self.c_swath), numberAzimuthLooks, numberRangeLooks
        )
        return pyRadarGridParameters.cbind(radarGrid)

# end of file
