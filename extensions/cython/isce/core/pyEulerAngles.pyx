#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Cartesian cimport cartesian_t, cartmat_t
from EulerAngles cimport EulerAngles
import numpy as np
cimport numpy as np


cdef class pyEulerAngles:
    '''
    Python wrapper for isce::core::EulerAngles

    Args:
        yaw (float): Yaw angle in radians
        pitch (float): Pitch angle in radians
        roll (float): Roll angle in radians
        yaw_orientation (Optional[str]): Can be either 'normal' or 'center'
    '''

    cdef EulerAngles * c_eulerangles
    cdef bool __owner

    def __cinit__(self,
                  np.ndarray[np.float64_t, ndim=1] time,
                  np.ndarray[np.float64_t, ndim=1] yaw,
                  np.ndarray[np.float64_t, ndim=1] pitch,
                  np.ndarray[np.float64_t, ndim=1] roll,
                  yaw_orientation='normal'):

        # Copy data to vectors manually (only doing this once, so hopefully
        # performance hit isn't too big of an issue)
        cdef i
        cdef int n = yaw.shape[0]
        cdef vector[double] vtime = vector[double](n)
        cdef vector[double] vyaw = vector[double](n)
        cdef vector[double] vpitch = vector[double](n)
        cdef vector[double] vroll = vector[double](n)
        for i in range(n):
            vtime[i] = time[i]
            vyaw[i] = yaw[i]
            vpitch[i] = pitch[i]
            vroll[i] = roll[i]
        
        # Instantiate EulerAngles object
        self.c_eulerangles = new EulerAngles(vtime, vyaw, vpitch, vroll,
            pyStringToBytes(yaw_orientation))
        self.__owner = True
        
    def __dealloc__(self):
        if self.__owner: 
            del self.c_eulerangles

    def ypr(self, double t):
        '''
        Return yaw, pitch and roll euler angles at a given time.

        Returns:
            np.array(3): euler angles
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
            sequence (list(3)): Sequence of angles. Example: 'ypr'

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
            ndarray[float]
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
            ndarray[float]
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
            ndarray[float]
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
    

# end of file
