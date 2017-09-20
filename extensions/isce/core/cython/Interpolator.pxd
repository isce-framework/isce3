#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector

cdef extern from "Interpolator.h" namespace "isce::core":
    cdef cppclass Interpolator:
        Interpolator() except +
        @staticmethod
        U bilinear[U](double,double,vector[vector[U]]&)
        @staticmethod
        U bicubic[U](double,double,vector[vector[U]]&)
        @staticmethod
        void sinc_coef(double,double,int,double,int,int&,int&,vector[double]&)
        @staticmethod
        U sinc_eval[U,V](vector[U]&,vector[V]&,int,int,int,double,int)
        @staticmethod
        U sinc_eval_2d[U,V](vector[vector[U]]&,vector[V]&,int,int,int,int,double,double,int,int)
        @staticmethod
        float interp_2d_spline(int,int,int,vector[vector[float]]&,double,double)
        @staticmethod
        double quadInterpolate(vector[double]&,vector[double]&,double)
        @staticmethod
        double akima(int,int,vector[vector[float]]&,double,double)

