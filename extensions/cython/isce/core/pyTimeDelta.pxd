#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017-2018
#

from TimeDelta cimport TimeDelta

cdef class pyTimeDelta:
    cdef TimeDelta c_timedelta

# end of file
