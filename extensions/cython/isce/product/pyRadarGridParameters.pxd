#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from RadarGridParameters cimport RadarGridParameters

cdef class pyRadarGridParameters:
    cdef RadarGridParameters * c_radargrid
    cdef bool __owner

