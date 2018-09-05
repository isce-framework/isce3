#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Metadata cimport Metadata

cdef class pyMetadata:
    """
    Cython wrapper for isce::product::Metadata.

    Args:
        None

    Return:
        None
    """
    # C++ class
    cdef Metadata c_metadata
    
    def __cinit__(self):
        """
        Constructor instantiates a C++ object and saves to python.
        """
        self.c_metadata = Metadata()

    @property
    def orbitNOE(self):
        """
        Get NOE orbit.
        """
        orbit = pyOrbit.cbind(self.c_metadata.orbitNOE())
        return orbit

    @orbitNOE.setter
    def orbitNOE(self, pyOrbit orbit):
        """
        Set NOE orbit.
        """
        self.c_metadata.orbitNOE(deref(orbit.c_orbit))

    @property
    def orbitPOE(self):
        """
        Get POE orbit.
        """
        orbit = pyOrbit.cbind(self.c_metadata.orbitPOE())
        return orbit

    @orbitPOE.setter
    def orbitPOE(self, pyOrbit orbit):
        """
        Set POE orbit.
        """
        self.c_metadata.orbitPOE(deref(orbit.c_orbit))

    @property
    def instrument(self):
        """
        Get radar instrument.
        """
        radar = pyRadar.cbind(self.c_metadata.instrument())
        return radar

    @instrument.setter
    def instrument(self, pyRadar instrument):
        """
        Set radar instrument.
        """
        self.c_metadata.instrument(deref(instrument.c_radar))

    @property
    def identification(self):
        """
        Get identification object.
        """
        pyID = pyIdentification.cbind(self.c_metadata.identification())
        return pyID

    @identification.setter
    def identification(self, pyIdentification pyID):
        """
        Set identification data.
        """
        self.c_metadata.identification(pyID.c_identification)
   
# end of file 
