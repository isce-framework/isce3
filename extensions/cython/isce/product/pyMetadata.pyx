#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2019
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
    cdef Metadata * c_metadata
    cdef bool __owner

    # Cython class members
    cdef pyOrbit py_orbit
    cdef pyEulerAngles py_attitude
    cdef pyProcessingInformation py_procInfo
    
    def __cinit__(self):
        """
        Constructor instantiates a C++ object and saves to python.
        """
        # Create the C++ Metadata class
        self.c_metadata = new Metadata()
        self.__owner = True

        # Bind the C++ Orbit class to the Cython pyOrbit instance
        self.py_orbit.c_orbit = &self.c_metadata.orbit()
        self.py_orbit.__owner = False

        # Bind the C++ EulerAngles class to the Cython pyEulerAngles instance
        self.py_attitude.c_eulerangles = &self.c_metadata.attitude()
        self.py_attitude.__owner = False

        # Bind the C++ ProcessingInformation class to the Cython pyProcessingInformation instance
        self.py_procInfo.c_procinfo = &self.c_metadata.procInfo()
        self.py_procInfo.__owner = False

    def __dealloc__(self):
        if self.__owner:
            del self.c_metadata

    @staticmethod
    def bind(pyMetadata meta):
        """
        Creates a new pyMetadata instance with C++ Metadata attribute shallow copied from
        another C++ Metadata attribute contained in a separate instance.

        Args:
            meta (pyMetadata): External pyMetadata instance to get C++ Metadata from.

        Returns:
            new_meta (pyMetadata): New pyMetadata instance with a shallow copy of C++ Metadata.
        """
        new_meta = pyMetadata()
        del new_meta.c_metadata
        new_meta.c_metadata = meta.c_metadata
        new_meta.__owner = False
        return new_meta

    @property
    def orbit(self):
        """
        Get orbit.
        """
        new_orbit = pyOrbit.bind(self.py_orbit)
        return new_orbit

    @orbit.setter
    def orbit(self, pyOrbit orb):
        """
        Set orbit.
        """
        self.c_metadata.orbit(deref(orb.c_orbit))

    @property
    def attitude(self):
        """
        Get Euler angles attitude.
        """
        new_attitude = pyEulerAngles.bind(self.py_attitude)
        return new_attitude

    @attitude.setter
    def attitude(self, pyEulerAngles euler):
        """
        Set Euler angles attitude.
        """
        self.c_metadata.attitude(deref(euler.c_eulerangles))

    @property
    def procInfo(self):
        """
        Get processing information.
        """
        new_proc = pyProcessingInformation.bind(self.py_procInfo)
        return new_proc
   
# end of file 
