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

    def __cinit__(self, list q):
        cdef vector[double] _q;
        for ii in range(4):
            _q.push_back(q[ii])
        self.c_quaternion = new Quaternion(_q)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_quaternion

    def ypr(self):
        '''
        Return Euler Angles corresponding to quaternions.

        Returns:
            numpy.array(3)
        '''
        cdef cartesian_t _ypr
        _ypr = self.c_quaternion.ypr()
        angles = np.asarray(<double[:3]>(&(_ypr[0])))
        return angles

    def factoredYPR(self, list position, list velocity, pyEllipsoid pyEllps):
        '''
        Returned Factored Euler Angles.
        '''
        cdef cartesian_t xyz
        cdef cartesian_t vel
        cdef int ii
        for ii in range(3):
            xyz[ii] = position[ii]
            vel[ii] = velocity[ii]
        cdef cartesian_t ypr_vec = self.c_quaternion.factoredYPR(xyz, vel, pyEllps.c_ellipsoid)
        angles = np.asarray(<double[:3]>(&(ypr_vec[0])))
        return angles

    def rotmat(self):
        '''
        Return the rotation matrix corresponding to the quaternions.

        Returns:
            numpy.array((3,3))
        '''
    
        cdef cartmat_t Rvec
        cdef string sequence_str = pyStringToBytes("")
        Rvec = self.c_quaternion.rotmat(sequence_str)
        R = np.empty((3,3), dtype=np.double)
        cdef double[:,:] Rview = R
        for ii in range(3):
            for jj in range(3):
                 Rview[ii][jj] = Rvec[ii][jj]

        return R

    @property
    def qvec(self):
        '''
        Return the quaternions.

        Returns:
            numpy.array(4)
        '''
        cdef vector[double] qv = self.c_quaternion.qvec()
        q = np.asarray(<double[:4]> (qv.data()))
        return q

    @qvec.setter
    def qvec(self, vec):
        '''
        Set the quaternions.

        Args:
            vec (list or numpy.array(4)): Quaternions.
        '''
        cdef vector[double] qv
        for ii in range(4):
            qv.push_back(vec[ii])
        self.c_quaternion.qvec(qv) 

# end of file
