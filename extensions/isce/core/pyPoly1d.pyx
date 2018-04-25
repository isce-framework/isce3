#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Poly1d cimport Poly1d
import numpy as np

cdef class pyPoly1d:
    #Pointer to underlying C++ object
    cdef Poly1d *c_poly1d

    #Variable that tracks ownership of underlying C++ object
    cdef bool __owner

    #Constructor
    def __cinit__(self, order=-1, mean=0., norm=1.):
        self.c_poly1d = new Poly1d(order,mean,norm)
        self.__owner = True

    #Destructor
    def __dealloc__(self):
        if self.__owner:
            del self.c_poly1d

    #Method to bind python wrapper to existing C++ pointer
    @staticmethod
    def bind(pyPoly1d poly):
        new_poly = pyPoly1d()
        del new_poly.c_poly1d
        new_poly.c_poly1d = poly.c_poly1d
        new_poly.__owner = False
        return new_poly

    #Interface to polynomial order
    @property
    def order(self):
        return self.c_poly1d.order

    @order.setter
    def order(self, int a):
        if (a < 0):
            return
        self.c_poly1d.order = a
        self.c_poly1d.coeffs.resize(a+1)

    #Interface to polynomial mean
    @property
    def mean(self):
        return self.c_poly1d.mean

    @mean.setter
    def mean(self, double a):
        self.c_poly1d.mean = a

    #Interface to polynomial norm
    @property
    def norm(self):
        return self.c_poly1d.norm

    @norm.setter
    def norm(self, double a):
        self.c_poly1d.norm = a

    #Memory view of C++ vector member
    @property
    def coeffs(self):
        cdef ssize_t N = self.c_poly1d.coeffs.size()
        cdef double[::1] v = <double[:N]>self.c_poly1d.coeffs.data()
        view = v
        return np.asarray(view)

    def copy(self, poly):
        try:
            self.order = poly.order
            self.mean = poly.mean
            self.norm = poly.norm
            self.coeffs = poly.coeffs
        except:
            print("Error: Object passed in to copy is incompatible with object of type pyPoly1d.")

    #Interface to C++ Set function. 
    def setCoeff(self, int a, double b):
        self.c_poly1d.setCoeff(a,b)

    #Interface to C++ Get function.
    def getCoeff(self, int a):
        return self.c_poly1d.getCoeff(a)

    #Interface to C++ eval function.
    def eval(self, double a):
        return self.c_poly1d.eval(a)

    #Interface to C++ print function.
    def printPoly(self):
        self.c_poly1d.printPoly()

    #Numpy like interface
    def __call__(self, x):
        return self.eval(x)

