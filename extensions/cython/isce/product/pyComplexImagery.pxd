#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from ComplexImagery cimport ComplexImagery

cdef class pyComplexImagery:
    cdef ComplexImagery c_compleximagery
       
# end of file 
