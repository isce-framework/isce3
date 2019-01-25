#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from ImageMode cimport ImageMode
from libcpp cimport bool

cdef class pyImageMode:
    cdef ImageMode * c_imagemode
    cdef bool __owner

    @staticmethod
    cdef cbind(ImageMode)
    
# end of file
