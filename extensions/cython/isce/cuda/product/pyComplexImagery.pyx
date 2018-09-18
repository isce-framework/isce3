#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from ComplexImagery cimport ComplexImagery

cdef class pyComplexImagery:
    """
    Cython wrapper for isce::product::ComplexImagery.

    Args:
        None

    Return:
        None
    """
    # C++ class
    cdef ComplexImagery c_compleximagery
    
    def __cinit__(self):
        """
        Constructor instantiates a C++ object and saves to python.
        """
        self.c_compleximagery = ComplexImagery()

    @property
    def auxMode(self):
        """
        Get the auxiliary ImageMode.
        """
        mode = pyImageMode.cbind(self.c_compleximagery.auxMode())
        return mode

    @auxMode.setter
    def auxMode(self, pyImageMode mode):
        """
        Set the auxiliary ImageMode.
        """
        self.c_compleximagery.auxMode(deref(mode.c_imagemode))

    @property
    def primaryMode(self):
        """
        Get the primary ImageMode.
        """
        mode = pyImageMode.cbind(self.c_compleximagery.primaryMode())
        return mode

    @primaryMode.setter
    def primaryMode(self, pyImageMode mode):
        """
        Set the primary ImageMode.
        """
        self.c_compleximagery.primaryMode(deref(mode.c_imagemode))
       
# end of file 
