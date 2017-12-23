#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector

cdef extern from "isce/core/LinAlg.h" namespace "isce::core":
    cdef cppclass LinAlg:
        LinAlg() except +
        @staticmethod
        void cross(vector[double]&,vector[double]&,vector[double]&)
        @staticmethod
        double dot(vector[double]&,vector[double]&)
        @staticmethod
        void linComb(double,vector[double]&,double,vector[double]&,vector[double]&)
        @staticmethod
        void matMat(vector[vector[double]]&,vector[vector[double]]&,vector[vector[double]]&)
        @staticmethod
        void matVec(vector[vector[double]]&,vector[double]&,vector[double]&)
        @staticmethod
        double norm(vector[double]&)
        @staticmethod
        void tranMat(vector[vector[double]]&,vector[vector[double]]&)
        @staticmethod
        void unitVec(vector[double]&,vector[double]&)
        @staticmethod
        void enuBasis(double,double,vector[vector[double]]&)
