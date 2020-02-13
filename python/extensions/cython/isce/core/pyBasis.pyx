#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Basis cimport Basis
from Cartesian cimport cartesian_t
import numpy as np
cimport numpy as np


cdef cartesian_t toVec3(x):
    assert len(x) == 3
    cdef cartesian_t cx
    cdef int ii
    for ii in range(3):
        cx[ii] = x[ii]
    return cx


cdef cartesian_t assertUnit(cartesian_t x, double tol=1e-8):
    cdef int ii
    cdef double xsum = 0.0
    for ii in range(3):
        xsum += x[ii]**2
    assert abs(xsum - 1.0) < tol, 'Input basis vector not a unit vector'
    return x


cdef class pyBasis:
    '''
    Python wrapper for isce::core::Basis
    '''

    cdef Basis c_basis

    def __cinit__(self, *args):
        """
        Basis()
        Basis(position, velocity) -> TCN
        Basis(x0, x1, x2)
        """
        if len(args) == 2:
            self.c_basis = Basis(toVec3(args[0]), toVec3(args[1]))
        else:
            self.c_basis = Basis()
            if len(args) == 3:
                self.x0(args[0])
                self.x1(args[1])
                self.x2(args[2])

    @property
    def x0(self):
        '''
        Return the first basis vector.

        Returns:
            numpy.array(3)
        '''
        #Make a memory view and copy
        cdef cartesian_t x0 = self.c_basis.x0()
        res = np.asarray((<double[:3]>(&x0[0])).copy())
        return res

    @x0.setter
    def x0(self, x):
        '''
        Set the first basis vector.

        Args:
            x (list or numpy.array(3)): list of floats
        '''
        self.c_basis.x0(assertUnit(toVec3(x)))

    @property
    def x1(self):
        '''
        Return the second basis vector.

        Returns:
            numpy.array(3)
        '''
        #Make a memory view and copy
        cdef cartesian_t x1 = self.c_basis.x1()
        res = np.asarray((<double[:3]>(&x1[0])).copy())
        return res

    @x1.setter
    def x1(self, x):
        '''
        Set the second basis vector.

        Args:
            x (list or numpy.array(3)): list of floats
        '''
        self.c_basis.x1(assertUnit(toVec3(x)))

    @property
    def x2(self):
        '''
        Return the third basis vector.

        Returns:
            numpy.array(3)
        '''
        #Make a memory view and copy
        cdef cartesian_t x2 = self.c_basis.x2()
        res = np.asarray((<double[:3]>(&x2[0])).copy())
        return res
    
    @x2.setter
    def x2(self, x):
        '''
        Set the third basis vector.

        Args:
            x (list or numpy.array(3)): list of floats
        '''
        self.c_basis.x2(assertUnit(toVec3(x)))
