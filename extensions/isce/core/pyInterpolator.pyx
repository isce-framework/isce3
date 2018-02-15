#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
#from libcpp.complex cimport complex
from Interpolator cimport Interpolator

cdef class pyInterpolator:
    cdef Interpolator *c_interp
    cdef bool __owner

    def __cinit__(self):
        self.c_interp = new Interpolator()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_interp
    # Note no static binder since we'll never need to pass any particular Interpolator object
    # around...
    '''
    def bilinear(self, double a, double b, c):
        cdef vector[vector[double]] _c0 = c
        cdef vector[vector[complex]] _c1 = c
        if type(c[0][0]) == type(complex()):
            return self.c_interp.bilinear[complex](a,b,_c1)
        else:
            return self.c_interp.bilinear[double](a,b,_c0)
    def bicubic(self, double a, double b, c):
        cdef vector[vector[double]] _c0 = c
        cdef vector[vector[complex]] _c1 = c
        if type(c[0][0]) == type(complex()):
            return self.c_interp.bicubic[complex](a,b,_c1)
        else:
            return self.c_interp.bicubic[double](a,b,_c0)
    '''
    def bilinear(self, double a, double b, c):
        cdef vector[vector[double]] _c = c
        return self.c_interp.bilinear[double](a,b,_c)
    def bicubic(self, double a, double b, c):
        cdef vector[vector[double]] _c = c
        return self.c_interp.bicubic[double](a,b,_c)
    def sinc_coef(self, double a, double b, int c, double d, int e, f, g, h=None):
        cdef int _f
        cdef int _g
        cdef vector[double] _h
        if (h):
            print("Error: Python does not allow for pass-by-reference, therefore the function " +
                  "call is modified.")
            print("       Please pass the pair of reference arguments (sixth and seventh " +
                  "position) as a tuple,")
            print("       and the 0th and 1st element of the tuple will contain the modified " +
                  "return value.")
            return
        _f = 0
        _g = 0
        for elem in g:
            _h.push_back(elem)
        self.c_interp.sinc_coef(a,b,c,d,e,_f,_g,_h)
        f[0] = _f
        f[1] = _g
    '''
    def sinc_eval(self, a, b, int c, int d, int e, double f):
        cdef vector[complex] _a0 = a
        cdef vector[double] _a1 = a
        cdef vector[double] _b = b
        if type(a[0]) == type(complex()):
            return self.c_interp.sinc_eval[complex,double](_a0,_b,c,d,e,f)
        else:
            return self.c_interp.sinc_eval[double,double](_a1,_b,c,d,e,f)
    def sinc_eval_2d(self, a, b, int c, int d, int e, int f, double g, double h, int i, int j):
        cdef vector[vector[complex]] _a0 = a
        cdef vector[vector[double]] _a1 = a
        cdef vector[double] _b = b
        if type(a[0][0]) == type(complex()):
            return self.c_interp.sinc_eval_2d[complex,double](_a0,_b,c,d,e,f,g,h,i,j)
        else:
            return self.c_interp.sinc_eval_2d[double,double](_a1,_b,c,d,e,f,g,h,i,j)
    '''
    def sinc_eval(self, a, b, int c, int d, int e, double f, int g):
        cdef vector[double] _a = a
        cdef vector[double] _b = b
        return self.c_interp.sinc_eval[double,double](_a,_b,c,d,e,f,g)
    def sinc_eval_2d(self, a, b, int c, int d, int e, int f, double g, double h, int i, int j):
        cdef vector[vector[double]] _a = a
        cdef vector[double] _b = b
        return self.c_interp.sinc_eval_2d[double,double](_a,_b,c,d,e,f,g,h,i,j)
    def interp_2d_spline(self, int a, int b, int c, list d, double e, double f):
        cdef vector[vector[float]] _d = d
        return self.c_interp.interp_2d_spline(a,b,c,_d,e,f)
    def quadInterpolate(self, a, b, double c):
        cdef vector[double] _a = a
        cdef vector[double] _b = b
        return self.c_interp.quadInterpolate(_a,_b,c)
    def akima(self, int a, int b, c, double d, double e):
        cdef vector[vector[float]] _c = c
        return self.c_interp.akima(a,b,_c,d,e)

