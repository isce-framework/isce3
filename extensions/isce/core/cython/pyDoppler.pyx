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
        self.c_doppler = new Doppler(
            deref(orbit.c_orbit),
            eulerangles.c_eulerangles,
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

        # Cache the old attitude values
        yaw_ref = self.eulerangles.yaw
        pitch_ref = self.eulerangles.pitch

        # Loop over range values
        for j in range(nr):

            # Yaw positive
            self.eulerangles.yaw = yaw_ref + dx
            fd_pos = self.centroid(slantRange[j], wvl, max_iter)

            # Yaw negative
            self.eulerangles.yaw = yaw_ref - dx
            fd_neg = self.centroid(slantRange[j], wvl, max_iter)

            # Yaw derivative
            outDerivs[j,0] = (fd_pos - fd_neg) / (2.0 * dx)
            # Reset
            self.eulerangles.yaw = yaw_ref

            # Pitch positive
            self.eulerangles.pitch = pitch_ref + dx
            fd_pos = self.centroid(slantRange[j], wvl, max_iter)

            # Pitch negative
            self.eulerangles.pitch = pitch_ref - dx
            fd_neg = self.centroid(slantRange[j], wvl, max_iter)

            # Pitch derivative
            outDerivs[j,1] = (fd_pos - fd_neg) / (2.0 * dx)
            # Reset
            self.eulerangles.pitch = pitch_ref

        return outDerivs
        

cdef class pyDopplerQuaternion(pyDoppler):
    cdef pyQuaternion quaternion
    
    def __cinit__(self, pyOrbit orbit, pyQuaternion quaternion, pyEllipsoid ellipsoid,
        double epoch):
        self.c_doppler = new Doppler(
            deref(orbit.c_orbit),
            quaternion.c_quaternion,
            deref(ellipsoid.c_ellipsoid), epoch
        )
        self.quaternion = quaternion
        self.__owner = True

    def derivs(self, np.ndarray[np.float64_t, ndim=1] slantRange, double wvl,
        string frame, int max_iter, int side, bool precession,
        np.ndarray[np.float64_t, ndim=2] outDerivs):
         
        cdef int nr, j, k
        cdef double fd_pos, fd_neg
        cdef double dx = 1.0e-10

        nr = slantRange.shape[0]

        # Cache the old attitude values
        cdef vector[double] qref = self.quaternion.c_quaternion.getQvec()

        # Loop over range values
        for j in range(nr):
            # Loop over quaternion elements
            for k in range(4):
                # Positive
                self.quaternion.c_quaternion.setQvecElement(qref[k] + dx, k)
                fd_pos = self.centroid(slantRange[j], wvl, frame, max_iter, side, precession)
                # Negative
                self.quaternion.c_quaternion.setQvecElement(qref[k] - dx, k)
                fd_neg = self.centroid(slantRange[j], wvl, frame, max_iter, side, precession)
                # Derivative
                outDerivs[j,k] = (fd_pos - fd_neg) / (2.0 * dx)
                # Reset
                self.quaternion.c_quaternion.setQvecElement(qref[k], k)

        return


# end of file
