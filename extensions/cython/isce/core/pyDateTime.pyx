#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
from cython.operator cimport dereference as deref
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from libcpp.string cimport string
from TimeDelta cimport TimeDelta
from DateTime cimport DateTime

cdef class pyDateTime:
    '''
    Python wrapper for isce::core::DateTime.
    Includes support for comparison and subtraction operators.

    Args:
        dt (:obj:`datetime.datetime` or :obj:`str`, optional): Input python datetime object or iso-8601 string
    '''
    cdef DateTime * c_datetime
    cdef bool __owner

    def __cinit__(self):
        '''
        Pre-constructor that creates a C++ isce::core::DateTime object and binds it to a python instance.
        '''
        self.c_datetime = new DateTime()
        self.__owner = True
   
    def __init__(self, dt=None):
        import datetime

        if isinstance(dt, (str, datetime.datetime)):
            self.set(dt)
        elif dt is not None:
            raise ValueError('pyDateTime object can be instantiated with a str or datetime.datetime object only')

    def set(self, dt):
        '''
        Set pyDateTime using datetime.datetime or str object.

        Args:
            dt (:obj:`datetime.datetime` or :obj:`str`): Input object.
        '''
        import datetime
        if isinstance(dt, str):
            self.strptime(dt)
        elif isinstance(dt, datetime.datetime):
            self.strptime(dt.isoformat())
        elif dt is not None:
            raise ValueError('pyDateTime object can be set with a str or datetime.datetime object only')


    def __dealloc__(self):
        if self.__owner:
            del self.c_datetime

    @staticmethod
    def bind(pyDateTime dt):
        '''
        Binds the current pyEllipsoid instance to another C++ DateTime pointer.

        Args:
            dt (:obj:`pyDateTime`): Source of C++ DateTime pointer.
        '''
        new_dt = pyDateTime()
        del new_dt.c_datetime
        new_dt.c_datetime = dt.c_datetime
        new_dt.__owner = False
        return new_dt

    @staticmethod
    cdef cbind(DateTime dt):
        '''
        Creates a new pyDateTime instance from a C++ DateTime instance.

        Args:
            dt (DateTime): C++ DateTime instance.
        '''
        new_dt = pyDateTime()
        del new_dt.c_datetime
        new_dt.c_datetime = new DateTime(dt)
        new_dt.__owner = True
        return new_dt

    def __richcmp__(self, pyDateTime dt, int comp):
        '''
        Rich comparison operator.
        
        Args:
            dt (:obj:`pyDateTime`): Time tag to compare against
            comp (int): Type of comparison
        '''
        if (comp == Py_LT):
            # <
            return deref(self.c_datetime) < deref(dt.c_datetime)
        elif (comp == Py_LE):
            # <=
            return deref(self.c_datetime) <= deref(dt.c_datetime)
        elif (comp == Py_EQ):
            # ==
            return deref(self.c_datetime) == deref(dt.c_datetime)
        elif (comp == Py_NE):
            # !=
            return deref(self.c_datetime) != deref(dt.c_datetime)
        elif (comp == Py_GT):
            # >
            return deref(self.c_datetime) > deref(dt.c_datetime)
        elif (comp == Py_GE):
            # >=
            return deref(self.c_datetime) >= deref(dt.c_datetime)
        else:
            assert False

    def __sub__(pyDateTime dt1, pyDateTime dt2):
        '''
        Time difference operator.
        
        Args:
            dt1 (:obj:`pyDateTime`): Time tag 1
            dt2 (:obj:`pyDateTime`): Time tag 2

        Returns:
            :obj:`pyTimeDelta`: Time difference between two pyDateTime tags.
        '''
        tdelta = pyTimeDelta()
        tdelta.c_timedelta = deref(dt1.c_datetime) - deref(dt2.c_datetime)
        return tdelta

    def __add__(pyDateTime dt1, pyTimeDelta delta):
        '''
        Addition operator.

        Args:
            dt1 (:obj:`pyDateTime`): Time tag
            delta (:obj:`pyDateTime`): Time difference to add

        Returns:
            :obj:`pyDateTime`: Resulting time tag
        '''
        return pyDateTime.cbind( deref(dt1.c_datetime) + delta.c_timedelta) 

    def isoformat(self):
        '''
        Return a string in ISO-8601 format.

        Returns:
            :obj:`str`: Date time in ISO-8601 format
        '''
        return self.c_datetime.isoformat().decode('UTF-8')

    def strptime(self, pydatestr):
        '''
        Sets underlying C++ DateTime object using time tag in ISO-8601 format

        Args:
            pydatestr (:obj:`str`): Time tag in ISO-8601 format
        '''
        self.c_datetime.strptime(pyStringToBytes(pydatestr))

# Instantiate a DateTime set at MIN_DATE_TIME
MIN_DATE_TIME = pyDateTime()
MIN_DATE_TIME.strptime('1970-01-01T00:00:00.000000000')

# end of file
