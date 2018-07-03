#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

import numpy as np
from Basis cimport Basis

cdef class pyBasis:

    cdef Basis c_basis

    def __cinit__(self):
        self.c_basis = Basis()

    @property
    def x0(self):
        return [self.c_basis.x0()[i] for i in range(3)]
    @x0.setter
    def x0(self, x):
        self._checklist(x)
        cdef cartesian_t cx
        cdef int i
        for i in range(3):
            cx[i] = x[i]
        self.c_basis.x0(cx)

    @property
    def x1(self):
        return [self.c_basis.x1()[i] for i in range(3)]
    @x1.setter
    def x1(self, x):
        self._checklist(x)
        cdef cartesian_t cx
        cdef int i
        for i in range(3):
            cx[i] = x[i]
        self.c_basis.x1(cx)

    @property
    def x2(self):
        return [self.c_basis.x2()[i] for i in range(3)]
    @x2.setter
    def x2(self, x):
        self._checklist(x)
        cdef cartesian_t cx
        cdef int i
        for i in range(3):
            cx[i] = x[i]
        self.c_basis.x2(cx)

    @staticmethod
    def _checklist(x):
        """
        Check the norm and length of the input list.
        """
        assert len(x) == 3, 'Input basis vector does not have three-elements'
        cdef int i
        cdef double xsum = 0.0
        for i in range(3):
            xsum += x[i]**2
        assert abs(xsum - 1.0) < 1.0e-8, 'Input basis vector not a unit vector'
        return

# end of file
