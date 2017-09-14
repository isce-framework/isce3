#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from LinAlg cimport LinAlg

cdef class pyLinAlg:
    cdef LinAlg *c_linAlg
    cdef bool __owner

    def __cinit__(self):
        self.c_linAlg = new LinAlg()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_linAlg
    # Note no static binder since we'll never need to pass any particular LinAlg object
    # around...

    def cross(self, list a, list b, list c):
        cdef vector[double] _a
        cdef vector[double] _b
        cdef vector[double] _c
        for i in range(3):
            _a.push_back(a[i])
            _b.push_back(b[i])
            _c.push_back(c[i])
        self.c_linAlg.cross(_a,_b,_c)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
            c[i] = _c[i]
    def dot(self, list a, list b):
        cdef vector[double] _a
        cdef vector[double] _b
        for i in range(3):
            _a.push_back(a[i])
            _b.push_back(b[i])
        return self.c_linAlg.dot(_a,_b)
    def linComb(self, double a, list b, double c, list d, list e):
        cdef vector[double] _b
        cdef vector[double] _d
        cdef vector[double] _e
        for i in range (3):
            _b.push_back(b[i])
            _d.push_back(d[i])
            _e.push_back(e[i])
        self.c_linAlg.linComb(a,_b,c,_d,_e)
        for i in range(3):
            b[i] = _b[i]
            d[i] = _d[i]
            e[i] = _e[i]
    def matMat(self, list a, list b, list c):
        cdef vector[vector[double]] _a
        cdef vector[vector[double]] _b
        cdef vector[vector[double]] _c
        cdef vector[double] tmp
        for i in range(3):
            tmp.push_back(0.)
        for i in range(3):
            _a.push_back(tmp)
            _b.push_back(tmp)
            _c.push_back(tmp)
        for i in range(3):
            for j in range(3):
                _a[i][j] = a[i][j]
                _b[i][j] = b[i][j]
                _c[i][j] = c[i][j]
        self.c_linAlg.matMat(_a,_b,_c)
        for i in range(3):
            for j in range(3):
                a[i][j] = _a[i][j]
                b[i][j] = _b[i][j]
                c[i][j] = _c[i][j]
    def matVec(self, list a, list b, list c):
        cdef vector[vector[double]] _a
        cdef vector[double] _b
        cdef vector[double] _c
        for i in range(3):
            _b.push_back(0.)
        for i in range(3):
            _a.push_back(_b)
        for i in range(3):
            for j in range(3):
                _a[i][j] = a[i][j]
            _b[i] = b[i]
            _c[i] = c[i]
        self.c_linAlg.matVec(_a,_b,_c)
        for i in range(3):
            for j in range(3):
                a[i][j] = _a[i][j]
            b[i] = _b[i]
            c[i] = _c[i]
    def norm(self, list a):
        cdef vector[double] _a
        for i in range(3):
            _a.push_back(a[i])
        return self.c_linAlg.norm(_a)
    def tranMat(self, list a, list b):
        cdef vector[vector[double]] _a
        cdef vector[vector[double]] _b
        cdef vector[double] tmp
        for i in range(3):
            tmp.push_back(0.)
        for i in range(3):
            _a.push_back(tmp)
            _b.push_back(tmp)
        for i in range(3):
            for j in range(3):
                _a[i][j] = a[i][j]
                _b[i][j] = b[i][j]
        self.c_linAlg.tranMat(_a,_b)
        for i in range(3):
            for j in range(3):
                a[i][j] = _a[i][j]
                b[i][j] = _b[i][j]
    def unitVec(self, list a, list b):
        cdef vector[double] _a
        cdef vector[double] _b
        for i in range(3):
            _a.push_back(a[i])
            _b.push_back(b[i])
        self.c_linAlg.unitVec(_a,_b)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
    def enuBasis(self, double a, double b, c):
        cdef vector[vector[double]] _c
        cdef vector[double] _temp
        for i in range(3):
            _temp.push_back(0)
        for i in range(3):
            _c.push_back(_temp)
        self.c_linAlg.enuBasis(a,b,_c)
        for i in range(3):
            for j in range(3):
                c[i][j] = _c[i][j]

