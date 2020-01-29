#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017-2018
#

from Projections cimport ProjectionBase, createProj
from Ellipsoid cimport Ellipsoid

cdef class pyProjection:
    '''
    Python wrapper for isce::core::ProjectionBase

    Args:
        epsg (int, epsg): EPSG code of projection system. Defaults to 4326.
    '''

    cdef ProjectionBase * c_proj

    def __cinit__(self):
        self.c_proj = NULL

    def __init__(self, epsg=4326):
        cdef int code = epsg
        self.c_proj = createProj(code)

    def ellipsoid(self):
        '''
        Get pyEllipsoid object associated with projection system
        '''
        
        cdef Ellipsoid elp = self.c_proj.ellipsoid()
        return pyEllipsoid(a=elp.a(), e2=elp.e2())

    def __dealloc__(self):
        del self.c_proj

    @property
    def EPSG(self):
        '''
        Return EPSG code
        '''
        return self.c_proj.code()

# end of file
