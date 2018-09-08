#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from TimeDelta cimport TimeDelta
from DateTime cimport DateTime

cdef class pyDateTime:
    cdef DateTime * c_datetime
    cdef bool __owner

    def __cinit__(self):
        self.c_datetime = new DateTime()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_datetime

    @staticmethod
    def bind(pyDateTime dt):
        new_dt = pyDateTime()
        del new_dt.c_datetime
        new_dt.c_datetime = dt.c_datetime
        new_dt.__owner = False
        return new_dt

    @staticmethod
    cdef cbind(DateTime dt):
        new_dt = pyDateTime()
        del new_dt.c_datetime
        new_dt.c_datetime = new DateTime(dt)
        new_dt.__owner = True
        return new_dt

    def __richcmp__(self, pyDateTime dt, int comp):
        if (comp == 0):
            # <
            return self.c_datetime < dt.c_datetime
        elif (comp == 1):
            # <=
            return self.c_datetime <= dt.c_datetime
        elif (comp == 2):
            # ==
            return self.c_datetime == dt.c_datetime
        elif (comp == 3):
            # !=
            return self.c_datetime != dt.c_datetime
        elif (comp == 4):
            # >
            return self.c_datetime > dt.c_datetime
        elif (comp == 5):
            # >=
            return self.c_datetime >= dt.c_datetime

    def __sub__(pyDateTime dt1, pyDateTime dt2):
        tdelta = pyTimeDelta()
        tdelta.c_timedelta = deref(dt1.c_datetime) - deref(dt2.c_datetime)
        return tdelta

    def isoformat(self):
        return str(self.c_datetime.isoformat())

    def strptime(self, pydatestr):
        self.c_datetime.strptime(pyStringToBytes(pydatestr))

# Instantiate a DateTime set at MIN_DATE_TIME
MIN_DATE_TIME = pyDateTime()
MIN_DATE_TIME.strptime('1970-01-01T00:00:00.000000000')

# end of file
