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
from Geo2rdr cimport *

cdef class pyGeo2rdr:
    cdef Geo2rdr * c_geo2rdr
    cdef bool __owner

    def __cinit__(self, pyEllipsoid ellipsoid, pyOrbit orbit, pyMetadata meta):
        self.c_geo2rdr = new Geo2rdr(
            deref(ellipsoid.c_ellipsoid),
            deref(orbit.c_orbit),
            meta.c_metadata
        )
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_geo2rdr

    def geo2rdr(self, pyRaster latRaster, pyRaster lonRaster, pyRaster hgtRaster,
                pyPoly2d doppler, outputDir, double azshift=0.0, double rgshift=0.0):
        """
        Run geo2rdr.
        """
        cdef string outdir = pyStringToBytes(outputDir)
        self.c_geo2rdr.geo2rdr(
            deref(latRaster.c_raster), deref(lonRaster.c_raster), deref(hgtRaster.c_raster),
            deref(doppler.c_poly2d), outdir, azshift, rgshift
        )

    def archive(self, metadata):
        load_archive[Geo2rdr](pyStringToBytes(metadata), 'Geo2rdr', self.c_geo2rdr)

# end of file
