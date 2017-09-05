#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#
# Note that this class is primarily meant to interface with numpy.datetime64. It has *very* limited
# functionality on the C++ side though

from libcpp cimport bool

cdef extern from "isce/core/DateTime.h" namespace "isce::core":
    cdef cppclass DateTime:
        #std::chrono::time_point<> t

        DateTime() except +
        DateTime(const double) except +
        
        bool operator==(const DateTime&)
        bool operator!=(const DateTime&)
        bool operator<(const DateTime&)
        bool operator<=(const DateTime&)
        bool operator>(const DateTime&)
        bool operator>=(const DateTime&)

        # Not supported yet
        #DateTime& operator+=(const double)
        #DateTime& operator-=(const double)

        double operator-(const DateTime&)

