#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from Serialization cimport load_archive
from Cartesian cimport cartesian_t
from Orbit cimport Orbit, orbitInterpMethod
import numpy as np
cimport numpy as np


cdef class pyOrbit:
    '''
    Python wrapper for isce::core::Orbit

    Args:
        basis (Optional[int]: 0 for SCH, 1 for WGS84
        nVectors (Optional [int]: Number of state vectors
    '''
    cdef Orbit *c_orbit
    cdef bool __owner

    def __cinit__(self, basis=1, nVectors=0):
        '''
        Pre-constructor that creates a C++ isce::core::Orbit object and binds it to python instance.
        '''
        self.c_orbit = new Orbit(basis,nVectors)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_orbit

    @staticmethod
    def bind(pyOrbit orb):
        new_orb = pyOrbit()
        del new_orb.c_orbit
        new_orb.c_orbit = orb.c_orbit
        new_orb.__owner = False
        return new_orb

    @property
    def basis(self):
        '''
        int: Basis code
        '''
        return self.c_orbit.basis

    @basis.setter
    def basis(self, int code):
        '''
        Set the basis code

        Args:
            a (int) : Value of basis code
        '''
        self.c_orbit.basis = code

    @property
    def nVectors(self):
        '''
        int: Number of state vectors.
        '''
        return self.c_orbit.nVectors


    @nVectors.setter
    def nVectors(self, int N):
        '''
        Set the number of state vectors.

        Args:
            N (int) : Number of state vectors.
        '''
        if (N < 0):
            raise ValueError('Number of state vectors cannot be < 0')

        self.c_orbit.nVectors = N
        self.c_orbit.UTCtime.resize(N)
        self.c_orbit.position.resize(3*N)
        self.c_orbit.velocity.resize(3*N)

    @property
    def UTCtime(self):
        '''
        list: UTC times corresponding to state vectors
        '''
        times = []
        cdef int ii
        for ii in range(self.nVectors):
            times.append(self.c_orbit.UTCtime[ii])

        return times

    @UTCtime.setter
    def UTCtime(self, times):
        '''
        Set the UTC times using a list or array.

        Args:
            times (list or np.array): UTC times corresponding to state vectors.
        '''
        if (self.nVectors != len(times)):
            raise ValueError("Invalid input size (expected list of length "+str(self.nVectors)+")")
        cdef int ii
        for ii in range(self.nVectors):
            self.c_orbit.UTCtime[ii] = times[ii]

    @property
    def position(self):
        '''
        np.array[nx3]: Array of positions corresponding to state vectors.
        '''
        pos = np.empty((self.nVectors,3), dtype=np.double)
        cdef double[:,:] posview = pos
        cdef int ii, jj
        for ii in range(self.nVectors):
            for jj in range(3):
                posview[ii,jj] = self.c_orbit.position[ii*3+jj]
        return pos

    @position.setter
    def position(self, a):
        if (3*self.nVectors != len(a)):
            print("Error: Invalid input size (expected list of length "+str(3*self.nVectors)+")")
            return
        for i in range(3*self.nVectors):
            self.c_orbit.position[i] = a[i]
    
    
    @property
    def velocity(self):
        a = []
        for i in range(3*self.nVectors):
            a.append(self.c_orbit.velocity[i])
        return a
    
    
    @velocity.setter
    def velocity(self, a):
        if (3*self.nVectors != len(a)):
            print("Error: Invalid input size (expected list of length "+str(3*self.nVectors)+")")
            return
        for i in range(3*self.nVectors):
            self.c_orbit.velocity[i] = a[i]
    
    def copy(self, orb):
        try:
            self.basis = orb.basis
            self.nVectors = orb.nVectors
            self.UTCtime = orb.UTCtime
            self.position = orb.position
            self.velocity = orb.velocity
        except:
            print("Error: Object passed in to copy is incompatible with object of type pyOrbit.")
    
    def dPrint(self):
        self.printOrbit()

    def getPositionVelocity(self, double a, list b, list c):
        cdef cartesian_t _b
        cdef cartesian_t _c
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        self.c_orbit.getPositionVelocity(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
    
    def getStateVector(self, int a, b, list c, list d):
        cdef cartesian_t _c
        cdef cartesian_t _d
        cdef double _b
        if (type(b) != type([])):
            print("Error: Python cannot pass primitives by reference.")
            print("To call this function, please pass the function an empty 1-tuple in the")
            print("second argument slot. The function will store the resulting time value")
            print("as the first (and only) element in the 1-tuple.")
        else:
            _b = 0.
            for i in range(3):
                _c[i] = c[i]
                _d[i] = d[i]
            self.c_orbit.getStateVector(a,_b,_c,_d)
            for i in range(3):
                c[i] = _c[i]
                d[i] = _d[i]
            b[0] = _b
    
    def setStateVector(self, int a, double b, list c, list d):
        cdef cartesian_t _c
        cdef cartesian_t _d
        for i in range(3):
            _c[i] = c[i]
            _d[i] = d[i]
        self.c_orbit.setStateVector(a,b,_c,_d)
        for i in range(3):
            c[i] = _c[i]
            d[i] = _d[i]
    
    def addStateVector(self, double a, list b, list c):
        cdef cartesian_t _b
        cdef cartesian_t _c
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        self.c_orbit.addStateVector(a,_b,_c)
    
    def interpolate(self, double a, list b, list c, int d):
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef orbitInterpMethod _d
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        if (d == orbitInterpMethod.HERMITE_METHOD):
            _d = orbitInterpMethod.HERMITE_METHOD
        elif (d == orbitInterpMethod.SCH_METHOD):
            _d = orbitInterpMethod.SCH_METHOD
        elif (d == orbitInterpMethod.LEGENDRE_METHOD):
            _d = orbitInterpMethod.LEGENDRE_METHOD
        else:
            print("Error: Unknown orbit interpolation method")
            return
        ret = self.c_orbit.interpolate(a,_b,_c,_d)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    
    def interpolateWGS84Orbit(self, double a, list b, list c):
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolateWGS84Orbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    
    def interpolateLegendreOrbit(self, double a, list b, list c):
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolateLegendreOrbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    
    def interpolateSCHOrbit(self, double a, list b, list c):
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
            _c[i] = c[i]
        ret = self.c_orbit.interpolateSCHOrbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    
    def computeAcceleration(self, double a, list b):
        cdef cartesian_t _b
        cdef int ret
        for i in range(3):
            _b[i] = b[i]
        ret = self.c_orbit.computeAcceleration(a,_b)
        for i in range(3):
            b[i] = _b[i]
        return ret
    
    def printOrbit(self):
        self.c_orbit.printOrbit()
    
    def loadFromHDR(self, a, int b=1):
        cdef bytes _a = a.encode()
        cdef char *cstring = _a
        self.c_orbit.loadFromHDR(cstring,b)
    
    def dumpToHDR(self, a):
        cdef bytes _a = a.encode()
        cdef char *cstring = _a
        self.c_orbit.dumpToHDR(cstring)

    def archive(self, metadata):
        load_archive[Orbit](pyStringToBytes(metadata),
                            'Orbit',
                            self.c_orbit)

