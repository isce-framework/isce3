#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from Position cimport Position

cdef class pyPosition:
    cdef Position *c_position
    cdef bool __owner

    def __cinit__(self):
        self.c_position = new Position()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_position
    @staticmethod
    def bind(pyPosition pos):
        new_pos = pyPosition()
        del new_pos.c_position
        new_pos.c_position = pos.c_position
        new_pos.__owner = True
        return new_pos
    
    @property
    def j(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_position.j[i]
        return a
    @j.setter
    def j(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_position.j[i] = a[i]
    @property
    def jdot(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_position.jdot[i]
        return a
    @jdot.setter
    def jdot(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_position.jdot[i] = a[i]
    @property
    def jddt(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_position.jddt[i]
        return a
    @jddt.setter
    def jddt(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_position.jddt[i] = a[i]
    def dPrint(self):
        print("J = "+str(self.j)+", jdot = "+str(self.jdot)+", jddt = "+str(self.jddt))
    def copy(self, ps):
        self.j = ps.j
        self.jdot = ps.jdot
        self.jddt = ps.jddt

    def lookVec(self, double a, double b, list c):
        cdef vector[double] _c
        for i in range(3):
            _c.push_back(c[i])
        self.c_position.lookVec(a,b,_c)
        for i in range(3):
            c[i] = _c[i]

