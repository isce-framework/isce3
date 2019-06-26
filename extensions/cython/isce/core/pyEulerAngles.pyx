#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Cartesian cimport cartesian_t, cartmat_t
from EulerAngles cimport EulerAngles, loadEulerAngles, saveEulerAngles
import numpy as np
cimport numpy as np
import h5py
from IH5 cimport hid_t, IGroup

cdef class pyEulerAngles:
    '''
    Python wrapper for isce::core::EulerAngles

    Args:
        yaw (float): Yaw angle in radians
        pitch (float): Pitch angle in radians
        roll (float): Roll angle in radians
        yaw_orientation (str, optional): Can be either 'normal' or 'center'
    '''

    cdef EulerAngles * c_eulerangles
    cdef bool __owner

    def __cinit__(self,
                  np.ndarray[np.float64_t, ndim=1] time=None,
                  np.ndarray[np.float64_t, ndim=1] yaw=None,
                  np.ndarray[np.float64_t, ndim=1] pitch=None,
                  np.ndarray[np.float64_t, ndim=1] roll=None,
                  yaw_orientation='normal'):

        # Copy data to vectors manually (only doing this once, so hopefully
        # performance hit isn't too big of an issue)
        cdef int i
        cdef int n 
        cdef vector[double] vtime
        cdef vector[double] vyaw
        cdef vector[double] vpitch
        cdef vector[double] vroll

        if time is not None and yaw is not None and pitch is not None and roll is not None:
            n = yaw.shape[0]
            vtime = vector[double](n)
            vyaw = vector[double](n)
            vpitch = vector[double](n)
            vroll = vector[double](n)
            for i in range(n):
                vtime[i] = time[i]
                vyaw[i] = yaw[i]
                vpitch[i] = pitch[i]
                vroll[i] = roll[i]
        
            # Instantiate EulerAngles object
            self.c_eulerangles = new EulerAngles(vtime, vyaw, vpitch, vroll,
                pyStringToBytes(yaw_orientation))

        else:
            self.c_eulerangles = new EulerAngles(pyStringToBytes(yaw_orientation))
        self.__owner = True
        
    def __dealloc__(self):
        if self.__owner: 
            del self.c_eulerangles

    @staticmethod
    def bind(pyEulerAngles euler):
        """
        Creates a new pyEulerAngles instance with C++ EulerAngles attribute shallow copied from
        another C++ EulerAngles attribute contained in a separate instance.

        Args:
            euler (pyEulerAngles): External pyEulerAngles instance to get C++ EulerAngles from.

        Returns:
            new_euler (pyEulerAngles): New pyEulerAngles instance with a shallow copy of 
                                       C++ EulerAngles.
        """
        new_euler = pyEulerAngles()
        del new_euler.c_eulerangles
        new_euler.c_eulerangles = euler.c_eulerangles
        new_euler.__owner = False
        return new_euler

    def ypr(self, double t):
        '''
        Return yaw, pitch and roll euler angles at a given time.

        Returns:
            numpy.array(3): Euler angles (ypr)
        '''
        cdef cartesian_t _ypr
        cdef double yaw = 0.0
        cdef double pitch = 0.0
        cdef double roll = 0.0
        self.c_eulerangles.ypr(t, yaw, pitch, roll)
        return yaw, pitch, roll

    def rotmat(self, double t, sequence):
        '''
        Return rotation matrix corresponding to angle sequence at a given time.

        Args:
            sequence (str): Sequence of angles. Example: 'ypr'

        Returns:
            numpy.array((3,3))
        '''

        cdef cartmat_t Rvec
        cdef string sequence_str = pyStringToBytes(sequence)
        Rvec = self.c_eulerangles.rotmat(t, sequence_str)
        R = np.empty((3,3), dtype=np.double)
        cdef double[:,:] Rview = R

        for ii in range(3):
            for jj in range(3):
                Rview[ii,jj] = Rvec[ii][jj]
        return R

    def quaternion(self, double t):
        '''
        Return quaternion representation of given euler angles at a given time.

        Returns:
            numpy.array(4)
        '''
        cdef vector[double] qvec = self.c_eulerangles.toQuaternionElements(t)
        res = np.asarray(<double[:4]>(&qvec[0]))
        return res

    @property
    def yaw(self):
        '''
        Return yaw angles in radians.

        Returns:
            numpy.ndarray[float]
        '''
        # Get vector of results
        cdef vector[double] values = self.c_eulerangles.yaw()
        cdef int n = values.size()

        # Copy back to numpy array
        cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros((n,), dtype=float)
        cdef int i
        for i in range(n):
            v[i] = values[i]
        return v

    @yaw.setter
    def yaw(self, value):
        raise NotImplementedError('Cannot set yaw values')

    @property
    def pitch(self):
        '''
        Return pitch angles in radians.

        Returns:
            numpy.ndarray[float]
        '''
        # Get vector of results
        cdef vector[double] values = self.c_eulerangles.pitch()
        cdef int n = values.size()

        # Copy back to numpy array
        cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros((n,), dtype=float)
        cdef int i
        for i in range(n):
            v[i] = values[i]
        return v

    @pitch.setter
    def pitch(self, value):
        raise NotImplementedError('Cannot set pitch values')

    @property
    def roll(self):
        '''
        Return roll angles in radians.

        Returns:
            numpy.ndarray[float]
        '''
        # Get vector of results
        cdef vector[double] values = self.c_eulerangles.roll()
        cdef int n = values.size()

        # Copy back to numpy array
        cdef np.ndarray[np.float64_t, ndim=1] v = np.zeros((n,), dtype=float)
        cdef int i
        for i in range(n):
            v[i] = values[i]
        return v

    @roll.setter
    def roll(self, value):
        raise NotImplementedError('Cannot set roll values')
    
    def loadFromH5(self, h5Group):
        '''
        Load EulerAngles from an HDF5 group

        Args:
            h5Group (h5py group): HDF5 group with Euler angles

        Returns:
            None
        '''

        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        loadEulerAngles(c_igroup, deref(self.c_eulerangles))

    def saveToH5(self, h5Group):
        '''
        Save EulerAngles to an HDF5 group

        Args:
            h5Group (h5py group): HDF5 group with Euler angles

        Returns:
            None
        '''

        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        saveEulerAngles(c_igroup, deref(self.c_eulerangles))
    

# end of file
