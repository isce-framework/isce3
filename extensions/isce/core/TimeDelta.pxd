#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram
# Copyright 2017-2018
#

from libcpp cimport bool

cdef extern from "isce/core/TimeDelta.h" namespace "isce::core":
    cdef cppclass TimeDelta:

        # Data members
        int days
        int hours
        int minutes
        int seconds
        double frac

        # Constructors
        TimeDelta() except +
        TimeDelta(double ss) except +
        TimeDelta(int hh, int mm, int ss) except +
        TimeDelta(int hh, int mm, double ss) except +
        TimeDelta(int hh, int mm, int ss, double ff) except +
        TimeDelta(int days, int hours, int minutes, int seconds, double frac) except +
        TimeDelta(const TimeDelta & ts) except +

        # Comparison operators
        bool operator<( const TimeDelta &ts)
        bool operator<( double ts)
        bool operator>( const TimeDelta &ts)
        bool operator>( double ts)
        bool operator<=( const TimeDelta &ts)
        bool operator<=( double ts)
        bool operator>=( const TimeDelta &ts)
        bool operator>=( double ts)
        bool operator==( const TimeDelta &ts)
        bool operator==( double ts)
        bool operator!=( const TimeDelta &ts)
        bool operator!=( double ts)

        # Math operators
        TimeDelta & operator=(const TimeDelta & ts)
        TimeDelta operator+(const TimeDelta & ts)
        TimeDelta operator+(const double & s)
        TimeDelta operator-(const TimeDelta & ts)
        TimeDelta operator-(const double & s)
        TimeDelta operator*(const double & s)
        TimeDelta operator/(const double & s)

        # Get methods
        double getTotalDays()
        double getTotalHours()
        double getTotalMinutes()
        double getTotalSeconds()

# end of file
