#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Peg cimport Peg

cdef class pyPeg:
    cdef Peg *c_peg
    cdef bool __owner

    def __cinit__(self, lat=0., lon=0., hdg=0.):
        self.c_peg = new Peg(lat,lon,hdg)
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_peg
    @staticmethod
    def bind(pyPeg peg):
        new_peg = pyPeg()
        del new_peg.c_peg
        new_peg.c_peg = peg.c_peg
        new_peg.__owner = False
        return new_peg

    @property
    def lat(self):
        return self.c_peg.lat
    @lat.setter
    def lat(self, double a):
        self.c_peg.lat = a
    @property
    def lon(self):
        return self.c_peg.lon
    @lon.setter
    def lon(self, double a):
        self.c_peg.lon = a
    @property
    def hdg(self):
        return self.c_peg.hdg
    @hdg.setter
    def hdg(self, double a):
        self.c_peg.hdg = a
    def dPrint(self):
        print("lat = "+str(self.lat)+", lon = "+str(self.lon)+", hdg = "+str(self.hdg))
    def copy(self, pg):
        try:
            self.lat = pg.lat
            self.lon = pg.lon
            self.hdg = pg.hdg
        # Note: this allows for a dummy class object to be passed in that just has lat, lon, and hdg 
        # as parameters!
        except:
            print("Error: Object passed in to copy is incompatible with object of type pyPeg.")
