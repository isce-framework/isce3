#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Metadata cimport Metadata

cdef class pyMetadata:
    cdef Metadata c_metadata
   
# end of file 
