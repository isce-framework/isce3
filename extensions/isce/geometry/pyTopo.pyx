#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2018
#

cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from cython.operator cimport dereference as deref

from SerializeGeometry cimport load_archive
from Topo cimport *

cdef class pyTopo:
    cdef Topo * c_topo
    cdef bool __owner

    def __cinit__(self, pyEllipsoid ellipsoid, pyOrbit orbit, pyMetadata meta):
        self.c_topo = new Topo(
            deref(ellipsoid.c_ellipsoid),
            deref(orbit.c_orbit),
            deref(meta.c_metadata)
        )
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_topo

    def topo(self, pyRaster demRaster, pyPoly2d doppler, outputDir):
        cdef string outdir = pyStringToBytes(outputDir)
        self.c_topo.topo(
            deref(demRaster.c_raster),
            deref(doppler.c_poly2d),
            outdir
        )

    def archive(self, metadata):
        load_archive[Topo](pyStringToBytes(metadata), 'Topo', self.c_topo)

# end of file
