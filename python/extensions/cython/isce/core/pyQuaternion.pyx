#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Cartesian cimport cartesian_t, cartmat_t
from Quaternion cimport Quaternion
import numpy as np
cimport numpy as np

cdef class pyQuaternion:
    '''
    Python wrapper for isce::core::Quaternion

    Args:
        q (list or np.array(4)): List of quaternions.
    '''

    cdef Quaternion * c_quaternion
    cdef bool __owner

    def __cinit__(self,
                  np.ndarray[np.float64_t, ndim=1] time,
                  np.ndarray[np.float64_t, ndim=2] quaternions):

        # Copy data to vectors manually (only doing this once, so hopefully
        # performance hit isn't too big of an issue)
        cdef i
        cdef int n = time.shape[0]
        cdef vector[double] vtime = vector[double](n)
        cdef vector[double] vquat = vector[double](n*4)
        for i in range(n):
            vtime[i] = time[i]
            for j in range(4):
                vquat[i*4+j] = quaternions[i,j]

        # Create Quaternion object
        self.c_quaternion = new Quaternion(vtime, vquat)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_quaternion

    def ypr(self, double t):
        '''
        Return Euler Angles corresponding to quaternions.

        Returns:
            numpy.array(3)
        '''
        cdef cartesian_t _ypr
        cdef double yaw = 0.0
        cdef double pitch = 0.0
        cdef double roll = 0.0
        self.c_quaternion.ypr(t, yaw, pitch, roll)
        return yaw, pitch, roll

    def factoredYPR(self, double t, list position, list velocity, pyEllipsoid pyEllps):
        '''
        Returned Factored Euler Angles.
        '''
        cdef cartesian_t xyz
        cdef cartesian_t vel
        cdef int ii
        for ii in range(3):
            xyz[ii] = position[ii]
            vel[ii] = velocity[ii]
        cdef cartesian_t ypr_vec = self.c_quaternion.factoredYPR(t, xyz, vel, pyEllps.c_ellipsoid)
        angles = np.asarray(<double[:3]>(&(ypr_vec[0])))
        return angles

    def rotmat(self, double t):
        '''
        Return the rotation matrix corresponding to the quaternions.

        Returns:
            numpy.array((3,3))
        '''
    
        cdef cartmat_t Rvec
        cdef string sequence_str = pyStringToBytes("")
        Rvec = self.c_quaternion.rotmat(t, sequence_str)
        R = np.empty((3,3), dtype=np.double)
        cdef double[:,:] Rview = R
        for ii in range(3):
            for jj in range(3):
                 Rview[ii][jj] = Rvec[ii][jj]

        return R
    
# end of file
