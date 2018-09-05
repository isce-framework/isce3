#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Identification cimport Identification

cdef class pyIdentification:
    """
    Cython wrapper for isce::product::Identification.

    Args:
        None

    Return:
        None
    """
    # C++ class
    cdef Identification c_identification
    
    def __cinit__(self):
        """
        Constructor instantiates a C++ object and saves to python.
        """
        self.c_identification = Identification()

    @staticmethod
    cdef cbind(Identification idobj):
        pyID = pyIdentification()
        pyID.c_identification = Identification(idobj)
        return pyID

    @property
    def lookDirection(self):
        """
        Get integer representing look direction.
        """
        return self.c_identification.lookDirection()

    @lookDirection.setter
    def lookDirection(self, value):
        """
        Set look direction from string or integer.
        """
        if isinstance(value, int):
            self.c_identification.lookDirection(<int> value)
        else:
            self.c_identification.lookDirection(<string> pyStringToBytes(value))

    @property
    def ellipsoid(self):
        """
        Get copy of ellipsoid.
        """
        ellps = pyEllipsoid.cbind(self.c_identification.ellipsoid())
        return ellps

    @ellipsoid.setter
    def ellipsoid(self, pyEllipsoid ellps):
        """
        Set ellipsoid.
        """
        self.c_identification.ellipsoid(deref(ellps.c_ellipsoid))

# end of file
