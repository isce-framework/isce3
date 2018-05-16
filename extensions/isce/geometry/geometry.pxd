#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram
# Copyright 2017-2018
#

from libcpp cimport bool

cdef extern from "isce/geometry/Geometry.h" namespace "isce::geometry":
    cdef cppclass Geometry:
        Geometry() except +
        
        @staticmethod
        int rdr2geo()

        

# end of file
