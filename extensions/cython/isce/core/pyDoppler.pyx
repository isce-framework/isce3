#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

cimport cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from Doppler cimport Doppler
from EulerAngles cimport Attitude, EulerAngles
from Quaternion cimport Quaternion

cdef class pyDoppler:

    cdef Doppler * c_doppler
    cdef int side
    cdef bool precession
    cdef string frame
    cdef bool __owner 

    def __cinit__(self):
        """
        Set __owner to False to prevent creation of base class
        """
        self.__owner = False 

    def __dealloc__(self):
        if self.__owner:
            del self.c_doppler

    def centroid(self, double slantRange, double wvl, int max_iter=10):
        cdef double fd = self.c_doppler.centroid(slantRange, wvl, self.frame, max_iter,
            self.side, self.precession)
        return fd

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def centroidProfile(self, np.ndarray[np.float64_t, ndim=1] slantRange,
        double wvl, int max_iter=10):

        cdef int i
        cdef int nr = slantRange.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] fd = np.zeros((nr,), dtype=slantRange.dtype)
        for i in range(nr):
            fd[i] = self.c_doppler.centroid(slantRange[i], wvl, self.frame, max_iter,
                self.side, self.precession) 
        return fd

    @property
    def satxyz(self):
        return [self.c_doppler.satxyz[i] for i in range(3)]
    @satxyz.setter
    def satxyz(self, value):
        raise ValueError('Cannot set satxyz')

    @property
    def satvel(self):
        return [self.c_doppler.satvel[i] for i in range(3)]
    @satvel.setter
    def satvel(self, value):
        raise ValueError('Cannot set satvel')

    @property
    def satllh(self):
        return [self.c_doppler.satllh[i] for i in range(3)]
    @satllh.setter
    def satllh(self, value):
        raise ValueError('Cannot set satllh')


cdef class pyDopplerEuler(pyDoppler):
    cdef pyEulerAngles eulerangles
    
    def __cinit__(self, pyOrbit orbit, pyEulerAngles eulerangles, pyEllipsoid ellipsoid,
        double epoch, int side=-1, bool precession=False, frame='inertial'):
        cdef Attitude*  attwrapper = <Attitude*>(eulerangles.c_eulerangles); 
        self.c_doppler = new Doppler(
            deref(orbit.c_orbit),
            attwrapper,
            deref(ellipsoid.c_ellipsoid), epoch
        )
        self.eulerangles = eulerangles
        self.side = side
        self.precession = precession
        self.frame = frame.encode('utf-8')
        self.__owner = True


    def derivs(self, np.ndarray[np.float64_t, ndim=1] slantRange, double wvl, int max_iter=10):
         
        cdef int nr, j
        cdef double fd_pos, fd_neg, yaw_ref, pitch_ref
        cdef double dx = 1.0e-10
        
        # Get number of range bins
        nr = slantRange.shape[0]

        # Allocate output derivatives
        cdef np.ndarray[np.float64_t, ndim=2] outDerivs = (
            np.zeros([nr,2], dtype=slantRange.dtype))

        # Loop over range values and compute derivatives
        cdef vector[double] vec_deriv
        for j in range(nr):
            vec_deriv = self.centroidDerivs(slantRange[j], wvl, max_iter)
            outDerivs[j,0] = vec_deriv[0]
            outDerivs[j,1] = vec_deriv[1]

        return outDerivs
        

cdef class pyDopplerQuaternion(pyDoppler):
    cdef pyQuaternion quaternion
    
    def __cinit__(self, pyOrbit orbit, pyQuaternion quaternion, pyEllipsoid ellipsoid,
        double epoch):
        cdef Attitude* quadwrapper = <Attitude*>(quaternion.c_quaternion)
        self.c_doppler = new Doppler(
            deref(orbit.c_orbit),
            quadwrapper,
            deref(ellipsoid.c_ellipsoid), epoch
        )
        self.quaternion = quaternion
        self.__owner = True

    def derivs(self, np.ndarray[np.float64_t, ndim=1] slantRange, double wvl,
        string frame, int max_iter, int side, bool precession):
         
        cdef int nr, j, k
        cdef double fd_pos, fd_neg
        cdef double dx = 1.0e-10

        # Get number of range bins
        nr = slantRange.shape[0]

        # Allocate output derivatives
        cdef np.ndarray[np.float64_t, ndim=2] outDerivs = (
            np.zeros([nr,4], dtype=slantRange.dtype))

        # Loop over range values and compute derivatives
        cdef vector[double] vec_deriv
        for j in range(nr):
            vec_deriv = self.centroidDerivs(slantRange[j], wvl, max_iter)
            for k in range(4):
                outDerivs[j,k] = vec_deriv[k]
                
        return outDerivs

# end of file
