#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
from TimeDelta cimport TimeDelta

cdef class pyTimeDelta:
    '''
    Python wrapper for isce::core::TimeDelta

    Args:
        inobj(:obj:`datetime.timedelta` or float, optional): Input python timedelta object or double precision floating point number
    '''
    cdef TimeDelta c_timedelta
    
    def __cinit__(self):
        '''
        Pre-constructor that creates a C++ isce::core::TimeDelta object and exposes it in Python.

        Note:
            There is no pointer binding for pyTimeDelta. This is meant to be ephemeral and not passed directly to C++ modules.
        '''
        self.c_timedelta = TimeDelta()

    def __init__(self, inobj=None):
        import datetime

        if isinstance(inobj, (int,float)):
            self.set(inobj)
        elif isinstance(inobj, datetime.timedelta):
            self.set(inobj)
        elif inobj is not None:
            raise ValueError('pyTimeDelta can only be initilized using datetime.timedelta or float')

    def set(self, inobj):
        '''
        Set the value using datetime.timedelta or float

        Args:
            inobj (:obj:`datetime.timedelta` or float): Input object
        '''
        import datetime
        if isinstance(inobj, (int,float)):
            self.c_timedelta = float(inobj)
        elif isinstance(inobj, datetime.timedelta):
            self.c_timedelta = float(inobj.total_seconds())
        else:
            raise ValueError('pyTimeDelta can only be set using datetime.timedelta or float')

    def getTotalDays(self):
        '''
        Equivalent number of days.

        Returns:
            float: Equivalent in number of days
        '''
        return self.c_timedelta.getTotalDays()

    def getTotalHours(self):
        '''
        Equivalent number of hours.

        Returns:
            float: Equivalent in number of hours
        '''
        return self.c_timedelta.getTotalHours()

    def getTotalMinutes(self):
        '''
        Equivalent number of minutes.

        Returns:
            float: Equivalent in number of minutes
        '''
        return self.c_timedelta.getTotalMinutes()

    def getTotalSeconds(self):
        '''
        Equivalent number of seconds.

        Returns:
            float: Equivalent in number of seconds.
        '''
        return self.c_timedelta.getTotalSeconds()

# end of file
