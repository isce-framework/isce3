#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from Peg cimport Peg

cdef class PyPeg:
    cdef Peg c_peg

    def __cinit__(self, lat=0., lon=0., hdg=0.):
        self.c_peg.lat = lat
        self.c_peg.lon = lon
        self.c_peg.hdg = hdg
    
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
        except: # Note: this allows for a dummy class object to be passed in that just has lat, lon, and hdg as parameters!
            print("Error: Object passed in to copy is incompatible with object of type PyPeg.")
