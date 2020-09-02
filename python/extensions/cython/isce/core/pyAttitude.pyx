#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Cartesian cimport cartesian_t, cartmat_t
from Attitude cimport Attitude, loadAttitude, saveAttitude
import numpy as np
cimport numpy as np
import h5py
from IH5 cimport hid_t, IGroup

cdef class pyAttitude:
    '''
    Python wrapper for isce3::core::Attitude
    '''

    cdef Attitude * c_attitude
    cdef bool __owner

    def __cinit__(self):
        self.c_attitude = new Attitude()
        self.__owner = True
        
    def __dealloc__(self):
        if self.__owner: 
            del self.c_attitude

    @staticmethod
    def bind(pyAttitude attitude):
        """
        Creates a new pyAttitude instance with C++ Attitude attribute shallow copied from
        another C++ Attitude attribute contained in a separate instance.

        Args:
            euler (pyAttitude): External pyAttitude instance to get C++ Attitude from.

        Returns:
            new_euler (pyAttitude): New pyAttitude instance with a shallow copy of 
                                       C++ Attitude.
        """
        new_attitude = pyAttitude()
        del new_attitude.c_attitude
        new_attitude.c_attitude = attitude.c_attitude
        new_attitude.__owner = False
        return new_attitude

    @staticmethod
    def loadFromH5(self, h5Group):
        '''
        Load Attitude from an HDF5 group

        Args:
            h5Group (h5py group): HDF5 group with attitude data

        Returns:
            pyAttitude object
        '''

        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        attitudeObj = pyAttitude()
        loadAttitude(c_igroup, deref(attitudeObj.c_attitude))

        return attitudeObj


    def saveToH5(self, h5Group):
        '''
        Save Attitude to an HDF5 group

        Args:
            h5Group (h5py group): HDF5 group with Euler angles

        Returns:
            None
        '''

        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        saveAttitude(c_igroup, deref(self.c_attitude))
    

# end of file
