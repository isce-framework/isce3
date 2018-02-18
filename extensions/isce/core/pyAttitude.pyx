#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Attitude cimport EulerAngles, Quaternion

cdef class pyEulerAngles:
    cdef EulerAngles * c_eulerangles
    cdef bool __owner

    def __cinit__(self, double yaw, double pitch, double roll, yaw_orientation='normal'):
        self.c_eulerangles = new EulerAngles(yaw, pitch, roll,
            yaw_orientation.encode('utf-8'))
        self.__owner = True
        
    def __dealloc__(self):
        if self.__owner: 
            del self.c_eulerangles

    def ypr(self):
        cdef vector[double] _ypr
        _ypr = self.c_eulerangles.ypr()
        angles = [_ypr[i] for i in range(3)]
        return angles

    def rotmat(self, sequence):
        cdef vector[vector[double]] Rvec
        cdef string sequence_str = sequence
        Rvec = self.c_eulerangles.rotmat(sequence_str)
        R = []
        for i in range(3):
            row = [Rvec[i][j] for j in range(3)]
            R.append(row)
        return R

    def quaternion(self):
        cdef vector[double] qvec = self.c_eulerangles.toQuaternionElements()
        q = [qvec[i] for i in range(4)]
        return q

    @property
    def yaw(self):
        return self.c_eulerangles.yaw()
    @yaw.setter
    def yaw(self, value):
        self.c_eulerangles.yaw(value)

    @property
    def pitch(self):
        return self.c_eulerangles.pitch()
    @pitch.setter
    def pitch(self, value):
        self.c_eulerangles.pitch(value)

    @property
    def roll(self):
        return self.c_eulerangles.roll()
    @roll.setter
    def roll(self, value):
        self.c_eulerangles.roll(value)


cdef class pyQuaternion:
    cdef Quaternion * c_quaternion
    cdef bool __owner

    def __cinit__(self, list q):
        cdef vector[double] _q;
        for i in range(4):
            _q.push_back(q[i])
        self.c_quaternion = new Quaternion(_q)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_quaternion

    def ypr(self):
        cdef vector[double] _ypr
        _ypr = self.c_quaternion.ypr()
        angles = [_ypr[i] for i in range(3)]
        return angles

    def factoredYPR(self, list position, list velocity, pyEllipsoid pyEllps):
        cdef vector[double] xyz
        cdef vector[double] vel
        cdef int i
        for i in range(3):
            xyz.push_back(position[i])
            vel.push_back(velocity[i])
        cdef vector[double] ypr_vec = self.c_quaternion.factoredYPR(xyz, vel, pyEllps.c_ellipsoid)
        angles = [ypr_vec[i] for i in range(3)]
        return angles

    def rotmat(self):
        cdef vector[vector[double]] Rvec
        cdef string sequence_str = "".encode('utf-8')
        Rvec = self.c_quaternion.rotmat(sequence_str)
        R = []
        for i in range(3):
            row = [Rvec[i][j] for j in range(3)]
            R.append(row)
        return R

    @property
    def qvec(self):
        cdef vector[double] qv = self.c_quaternion.qvec()
        q = [qv[i] for i in range(4)]
        return q
    @qvec.setter
    def qvec(self, vec):
        cdef vector[double] qv
        for i in range(4):
            qv.push_back(vec[i])
        self.c_quaternion.qvec(qv) 

# end of file
