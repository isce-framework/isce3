#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from Ellipsoid cimport Ellipsoid
from Serialization cimport load_archive

cdef class pyEllipsoid:
    cdef Ellipsoid *c_ellipsoid
    cdef bool __owner

    def __cinit__(self, a=0., e2=0.):
        self.c_ellipsoid = new Ellipsoid(a, e2)
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_ellipsoid
    @staticmethod
    def bind(pyEllipsoid elp):
        new_elp = pyEllipsoid()
        del new_elp.c_ellipsoid
        new_elp.c_ellipsoid = elp.c_ellipsoid
        new_elp.__owner = False
        return new_elp

    @property
    def a(self):
        return self.c_ellipsoid.a()
    @a.setter
    def a(self, double a):
        self.c_ellipsoid.a(a)
    @property
    def e2(self):
        return self.c_ellipsoid.e2()
    @e2.setter
    def e2(self, double a):
        self.c_ellipsoid.e2(a)
    def copyFrom(self, elp):
        # Replaces copy-constructor functionality
        try:
            self.a = elp.a
            self.e2 = elp.e2
        # Note: this allows for a dummy class object to be passed in that just has a and e2 as 
        # parameters!
        except: 
            print("Error: Object passed in to copy is incompatible with object of type " +
                  "pyEllipsoid.")

    def rEast(self, double a):
        return self.c_ellipsoid.rEast(a)
    def rNorth(self, double a):
        return self.c_ellipsoid.rNorth(a)
    def rDir(self, double a, double b):
        return self.c_ellipsoid.rDir(a,b)
    def latLonToXyz(self, list a, list b):
        cdef cartesian_t _a
        cdef cartesian_t _b
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
        self.c_ellipsoid.latLonToXyz(_a,_b)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
    def xyzToLatLon(self, list a, list b):
        cdef cartesian_t _a
        cdef cartesian_t _b
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
        self.c_ellipsoid.xyzToLatLon(_a,_b)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
    def getAngs(self, list a, list b, list c, d, e=None):
        cdef cartesian_t _a
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef double _d
        cdef double _e
        if (e):
            print("Error: Python cannot pass primitives by reference.")
            print("To call this function, please pass the function an empty tuple as the fourth")
            print("argument (no fifth argument). The first element of the list will be the azimuth")
            print("angle, the second element will be the look angle.")
            return
        else:
            _d = 0.
            _e = 0.
            for i in range(3):
                _a[i] = a[i]
                _b[i] = b[i]
                _c[i] = c[i]
            self.c_ellipsoid.getAngs(_a,_b,_c,_d,_e)
            for i in range(3):
                a[i] = _a[i]
                b[i] = _b[i]
                c[i] = _c[i]
            d[0] = _d
            d[1] = _e
    def getTCN_TCvec(self, list a, list b, list c, list d):
        cdef cartesian_t _a
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef cartesian_t _d
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
            _c[i] = c[i]
            _d[i] = d[i]
        self.c_ellipsoid.getTCN_TCvec(_a,_b,_c,_d)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
            c[i] = _c[i]
            d[i] = _d[i]
    def TCNbasis(self, list a, list b, list c, list d, list e):
        cdef cartesian_t _a
        cdef cartesian_t _b
        cdef cartesian_t _c
        cdef cartesian_t _d
        cdef cartesian_t _e
        for i in range(3):
            _a[i] = a[i]
            _b[i] = b[i]
            _c[i] = c[i]
            _d[i] = d[i]
            _e[i] = e[i]
        self.c_ellipsoid.TCNbasis(_a,_b,_c,_d,_e)
        for i in range(3):
            c[i] = _c[i]
            d[i] = _d[i]
            e[i] = _e[i]

    def archive(self, metadata):
        load_archive[Ellipsoid](pyStringToBytes(metadata),
                                'Ellipsoid',
                                self.c_ellipsoid)

