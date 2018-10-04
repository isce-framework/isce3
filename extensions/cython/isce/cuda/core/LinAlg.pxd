#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector
from Cartesian cimport cartesian_t, cartmat_t

cdef extern from "isce/core/LinAlg.h" namespace "isce::core":
    cdef cppclass LinAlg:
        LinAlg() except +
        @staticmethod
        void cross(cartesian_t&,cartesian_t&,cartesian_t&)
        @staticmethod
        double dot(cartesian_t&,cartesian_t&)
        @staticmethod
        void linComb(double,cartesian_t&,double,cartesian_t&,cartesian_t&)
        @staticmethod
        void matMat(cartmat_t&,cartmat_t&,cartmat_t&)
        @staticmethod
        void matVec(cartmat_t&,cartesian_t&,cartesian_t&)
        @staticmethod
        double norm(cartesian_t&)
        @staticmethod
        void tranMat(cartmat_t&,cartmat_t&)
        @staticmethod
        void unitVec(cartesian_t&,cartesian_t&)
        @staticmethod
        void enuBasis(double,double,cartmat_t&)
