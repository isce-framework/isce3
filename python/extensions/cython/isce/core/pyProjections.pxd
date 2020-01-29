#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2020
#

from Projections cimport ProjectionBase

cdef class pyProjection:
    cdef ProjectionBase * c_proj

