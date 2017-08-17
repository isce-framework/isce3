#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from Pegtrans cimport Pegtrans

cdef class PyPegtrans:
    cdef Pegtrans c_pegtrans

    def __cinit__(self): # Never will be initialized with values, so no need to check
        return
    
    @property
    def mat(self):
        a = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for i in range(3):
            for j in range(3):
                a[i][j] = self.c_pegtrans.mat[i][j]
        return a
    @mat.setter
    def mat(self, a):
        if ((len(a) != 3) or (len(a[0]) != 3)):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            for j in range(3):
                self.c_pegtrans.mat[i][j] = a[i][j]
    @property
    def matinv(self):
        a = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for i in range(3):
            for j in range(3):
                a[i][j] = self.c_pegtrans.matinv[i][j]
        return a
    @matinv.setter
    def matinv(self, a):
        if ((len(a) != 3) or (len(a[0]) != 3)):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            for j in range(3):
                self.c_pegtrans.matinv[i][j] = a[i][j]
    @property
    def ov(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_pegtrans.ov[i]
        return a
    @ov.setter
    def ov(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_pegtrans.ov[i] = a[i]
    @property
    def radcur(self):
        return self.c_pegtrans.radcur
    @radcur.setter
    def radcur(self, double a):
        self.c_pegtrans.radcur = a
    def dPrint(self):
        print("Mat = "+str(self.mat)+", matinv = "+str(self.matinv)+", ov = "+str(self.ov)+", radcur = "+str(self.radcur))
    def copy(self, pt):
        try:
            self.mat = pt.mat
            self.matinv = pt.matinv
            self.ov = pt.ov
            self.radcur = pt.radcur
        except:
            print("Error: Object passed in is incompatible with object of type PyPegtrans.")

    def radarToXYZ(self, PyEllipsoid a, PyPeg b):
        self.c_pegtrans.radarToXYZ(a.c_ellipsoid,b.c_peg)
    def convertSCHtoXYZ(self, list a, list b, int c):
        cdef vector[double] _a
        cdef vector[double] _b
        for i in range(3):
            _a.push_back(a[i])
            _b.push_back(b[i])
        self.c_pegtrans.convertSCHtoXYZ(_a,_b,c)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
    def convertSCHdotToXYZdot(self, list a, list b, list c, list d, int e):
        cdef vector[double] _a
        cdef vector[double] _b
        cdef vector[double] _c
        cdef vector[double] _d
        for i in range(3):
            _a.push_back(a[i])
            _b.push_back(b[i])
            _c.push_back(c[i])
            _d.push_back(d[i])
        self.c_pegtrans.convertSCHdotToXYZdot(_a,_b,_c,_d,e)
        for i in range(3):
            a[i] = _a[i]
            b[i] = _b[i]
            c[i] = _c[i]
            d[i] = _d[i]
    def SCHbasis(self, list a, list b, list c):
        cdef vector[double] _a
        cdef vector[double] _temp_b
        cdef vector[double] _temp_c
        cdef vector[vector[double]] _b
        cdef vector[vector[double]] _c
        for i in range(3):
            _a.push_back(a[i])
            _temp_b.push_back(0.)
            _temp_c.push_back(0.)
        for i in range(3):
            for j in range(3):
                _temp_b[j] = b[i][j]
                _temp_c[j] = c[i][j]
            _b.push_back(_temp_b)
            _c.push_back(_temp_c)
        self.c_pegtrans.SCHbasis(_a,_b,_c)
        for i in range(3):
            a[i] = _a[i]
            for j in range(3):
                b[i][j] = _b[i][j]
                c[i][j] = _c[i][j]

