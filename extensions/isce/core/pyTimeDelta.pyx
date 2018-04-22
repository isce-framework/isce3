#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
from TimeDelta cimport TimeDelta

cdef class pyTimeDelta:
    cdef TimeDelta c_timedelta
    
    def __cinit__(self):
        self.c_timedelta = TimeDelta()

    def getTotalDays(self):
        return self.c_timedelta.getTotalDays()

    def getTotalHours(self):
        return self.c_timedelta.getTotalHours()

    def getTotalMinutes(self):
        return self.c_timedelta.getTotalMinutes()

    def getTotalSeconds(self):
        return self.c_timedelta.getTotalSeconds()

# end of file
