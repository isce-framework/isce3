#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from Raster cimport Raster

cdef class pyRaster:
    cdef Raster * c_raster
    cdef bool __owner

# end of file        
