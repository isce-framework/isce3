#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from Orbit cimport Orbit, orbitInterpMethod
from libcpp.vector cimport vector

cdef class PyOrbit:
    cdef Orbit c_orbit

    def __cinit__(self, basis=1, nVectors=0):
        self.c_orbit.basis = basis
        self.c_orbit.nVectors = nVectors
        self.c_orbit.position.resize(3*nVectors)
        self.c_orbit.velocity.resize(3*nVectors)
        self.c_orbit.UTCtime.resize(nVectors)

    @property
    def basis(self):
        return self.c_orbit.basis
    @basis.setter
    def basis(self, int a):
        self.c_orbit.basis = a
    @property
    def nVectors(self):
        return self.c_orbit.nVectors
    @nVectors.setter
    def nVectors(self, int a):
        if (a < 0):
            return
        self.c_orbit.nVectors = a
        self.c_orbit.UTCtime.resize(a)
        self.c_orbit.position.resize(3*a)
        self.c_orbit.velocity.resize(3*a)
    @property
    def UTCtime(self):
        a = []
        for i in range(self.nVectors):
            a.append(self.c_orbit.UTCtime[i])
        return a
    @UTCtime.setter
    def UTCtime(self, a):
        if (self.nVectors != len(a)):
            print("Error: Invalid input size (expected list of length "+str(self.nVectors)+")")
            return
        for i in range(self.nVectors):
            self.c_orbit.UTCtime[i] = a[i]
    @property
    def position(self):
        a = []
        for i in range(3*self.nVectors):
            a.append(self.c_orbit.position[i])
        return a
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
            print("Error: Object passed in to copy is incompatible with object of type PyOrbit.")
    def dPrint(self):
        self.printOrbit()

    def getPositionVelocity(self, double a, list b, list c):
        cdef vector[double] _b
        cdef vector[double] _c
        for i in range(3):
            _b.push_back(b[i])
            _c.push_back(c[i])
        self.c_orbit.getPositionVelocity(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
    def getStateVector(self, int a, b, list c, list d):
        cdef vector[double] _c
        cdef vector[double] _d
        cdef double _b
        if (type(b) != type([])):
            print("Error: Python cannot pass primitives by reference.")
            print("To call this function, please pass the function an empty 1-tuple in the")
            print("second argument slot. The function will store the resulting time value")
            print("as the first (and only) element in the 1-tuple.")
        else:
            _b = 0.
            for i in range(3):
                _c.push_back(c[i])
                _d.push_back(d[i])
            self.c_orbit.getStateVector(a,_b,_c,_d)
            for i in range(3):
                c[i] = _c[i]
                d[i] = _d[i]
            b[0] = _b
    def setStateVector(self, int a, double b, list c, list d):
        cdef vector[double] _c
        cdef vector[double] _d
        for i in range(3):
            _c.push_back(c[i])
            _d.push_back(d[i])
        self.c_orbit.setStateVector(a,b,_c,_d)
        for i in range(3):
            c[i] = _c[i]
            d[i] = _d[i]
    def addStateVector(self, double a, list b, list c):
        cdef vector[double] _b
        cdef vector[double] _c
        for i in range(3):
            _b.push_back(b[i])
            _c.push_back(c[i])
        self.c_orbit.addStateVector(a,_b,_c)
    def interpolate(self, double a, list b, list c, int d):
        cdef vector[double] _b
        cdef vector[double] _c
        cdef orbitInterpMethod _d
        cdef int ret
        for i in range(3):
            _b.push_back(b[i])
            _c.push_back(c[i])
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
        cdef vector[double] _b
        cdef vector[double] _c
        cdef int ret
        for i in range(3):
            _b.push_back(b[i])
            _c.push_back(c[i])
        ret = self.c_orbit.interpolateWGS84Orbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def interpolateLegendreOrbit(self, double a, list b, list c):
        cdef vector[double] _b
        cdef vector[double] _c
        cdef int ret
        for i in range(3):
            _b.push_back(b[i])
            _c.push_back(c[i])
        ret = self.c_orbit.interpolateLegendreOrbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def interpolateSCHOrbit(self, double a, list b, list c):
        cdef vector[double] _b
        cdef vector[double] _c
        cdef int ret
        for i in range(3):
            _b.push_back(b[i])
            _c.push_back(c[i])
        ret = self.c_orbit.interpolateSCHOrbit(a,_b,_c)
        for i in range(3):
            b[i] = _b[i]
            c[i] = _c[i]
        return ret
    def computeAcceleration(self, double a, list b):
        cdef vector[double] _b
        cdef int ret
        for i in range(3):
            _b.push_back(b[i])
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

