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
    '''
    Python wrapper for isce::core::DateTime.
    Includes support for comparison and subtraction operators.

    Args:
        inobj (:obj:`datetime.datetime` or :obj:`str`, optional): Input python datetime object or iso-8601 string
    '''
    cdef DateTime * c_datetime
    cdef bool __owner

    def __cinit__(self):
        '''
        Pre-constructor that creates a C++ isce::core::DateTime object and binds it to a python instance.
        '''
        self.c_datetime = new DateTime()
        self.__owner = True
   
    def __init__(self, inobj=None):
        import datetime

        if isinstance(inobj, (str, datetime.datetime)):
            self.set(inobj)
        elif inobj is not None:
            raise ValueError('pyDateTime object can be instantiated with a str or datetime.datetime object only')

    def set(self, inobj):
        '''
        Set pyDateTime using datetime.datetime or str object.

        Args:
            inobj (:obj:`datetime.datetime` or :obj:`str`): Input object.
        '''
        import datetime
        if isinstance(inobj, str):
            self.strptime(inobj)
        elif isinstance(inobj, datetime.datetime):
            self.strptime(inobj.isoformat())
        elif inobj is not None:
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
