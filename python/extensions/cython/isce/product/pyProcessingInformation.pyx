#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

import numpy as np
cimport numpy as np
from Matrix cimport valarray
from LUT2d cimport LUT2d
from ProcessingInformation cimport ProcessingInformation

cdef class pyProcessingInformation:
    """
    Cython wrapper for isce::product::ProcessingInformation.

    Args:
        None

    Return:
        None
    """
    # C++ class pointers
    cdef ProcessingInformation * c_procinfo
    cdef bool __owner

    def __cinit__(self):
        """
        Constructor instantiates a C++ object and saves to python.
        """
        self.c_procinfo = new ProcessingInformation()
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_procinfo

    @staticmethod
    def bind(pyProcessingInformation proc):
        """
        Creates a new pyProcessingInformation instance with C++ ProcessingInformation
        attribute shallow copied from another C++ ProcessingInformation attribute contained
        in a separate instance.

        Args:
            proc (pyProcessingInformation): External pyProcessingInformation instance to
                                            get C++ ProcessingInformation from.

        Returns:
            new_proc (pyProcessingInformation): New pyProcessingInformation instance with a
                                                shallow copy of C++ ProcessingInformation.
        """
        new_proc = pyProcessingInformation()
        del new_proc.c_procinfo
        new_proc.c_procinfo = proc.c_procinfo
        new_proc.__owner = False
        return new_proc

    @property
    def slantRange(self):
        """
        Return slant range array.
        """
        cdef valarray[double] v = self.c_procinfo.slantRange()
        return valarrayToNumpy(v)

    @property
    def zeroDopplerTime(self):
        """
        Return zero Doppler time array.
        """
        cdef valarray[double] v = self.c_procinfo.zeroDopplerTime()
        return valarrayToNumpy(v)

    @property
    def effectiveVelocity(self):
        """
        Return look-up-table for effective velocity.
        """
        cdef LUT2d[double] lut = self.c_procinfo.effectiveVelocity()
        py_lut = pyLUT2d.cbind(lut)
        return py_lut

    def azimuthFMRate(self, freq='A'):
        """
        Get look-up-table for azimuth FM rate for a given center frequency.
        """
        cdef string freq_str = pyStringToBytes(freq)
        cdef LUT2d[double] lut = self.c_procinfo.azimuthFMRate(freq_str[0])
        py_lut = pyLUT2d.cbind(lut)
        return py_lut

    def dopplerCentroid(self, freq='A'):
        """
        Get look-up-table for Doppler centroid for a given center frequency.
        """
        cdef string freq_str = pyStringToBytes(freq)
        cdef LUT2d[double] lut = self.c_procinfo.dopplerCentroid(freq_str[0])
        py_lut = pyLUT2d.cbind(lut)
        return py_lut

# end of file
