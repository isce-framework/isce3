#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from Poly1d cimport Poly1d

cdef class PyPoly1d:
    cdef Poly1d c_poly1d

    def __cinit__(self, order=-1, mean=0., norm=1.):
        self.c_poly1d.order = order
        self.c_poly1d.mean = mean
        self.c_poly1d.norm = norm
        self.c_poly1d.coeffs.resize(order+1)

    @property
    def order(self):
        return self.c_poly1d.order
    @order.setter
    def order(self, int a):
        if (a < 0):
            return
        self.c_poly1d.order = a
        self.c_poly1d.coeffs.resize(a+1)
    @property
    def mean(self):
        return self.c_poly1d.mean
    @mean.setter
    def mean(self, double a):
        self.c_poly1d.mean = a
    @property
    def norm(self):
        return self.c_poly1d.norm
    @norm.setter
    def norm(self, double a):
        self.c_poly1d.norm = a
    @property
    def coeffs(self):
        a = []
        for i in range(self.order+1):
            a.append(self.c_poly1d.coeffs[i])
        return a
    @coeffs.setter
    def coeffs(self, a):
        if (self.order+1 != len(a)):
            print("Error: Invalid input size (expected list of length "+str(self.order+1)+")")
            return
        for i in range(self.order+1):
            self.c_poly1d.coeffs[i] = a[i]
    def copy(self, poly):
        try:
            self.order = poly.order
            self.mean = poly.mean
            self.norm = poly.norm
            self.coeffs = poly.coeffs
        except:
            print("Error: Object passed in to copy is incompatible with object of type PyPoly1d.")
    def dPrint(self):
        self.printPoly()

    def setCoeff(self, int a, double b):
        self.c_poly1d.setCoeff(a,b)
    def getCoeff(self, int a):
        return self.c_poly1d.getCoeff(a)
    def eval(self, double a):
        return self.c_poly1d.eval(a)
    def printPoly(self):
        self.c_poly1d.printPoly()

