#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from DateTime cimport DateTime

cdef class pyDateTime:
    cdef DateTime c_dateTime

    def __cinit__(self):
        return

    def __richcmp__(self, pyDateTime dt, int comp):
        if (comp == 0):
            # <
            return self.c_dateTime < dt.c_dateTime
        elif (comp == 1):
            # <=
            return self.c_dateTime <= dt.c_dateTime
        elif (comp == 2):
            # ==
            return self.c_dateTime == dt.c_dateTime
        elif (comp == 3):
            # !=
            return self.c_dateTime != dt.c_dateTime
        elif (comp == 4):
            # >
            return self.c_dateTime > dt.c_dateTime
        elif (comp == 5):
            # >=
            return self.c_dateTime >= dt.c_dateTime

    '''     NOT YET SUPPORTED IN CYTHON
    def __iadd__(self, const double time_delta):
        self.c_dateTime += time_delta

    def __isub__(self, const double time_delta):
        self.c_dateTime -= time_delta
    '''

    def __sub__(self, pyDateTime dt):
        return self.c_dateTime - dt.c_dateTime
