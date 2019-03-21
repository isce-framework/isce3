#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp cimport bool
from TimeDelta cimport TimeDelta

cdef extern from "isce/core/DateTime.h" namespace "isce::core":
    cdef cppclass DateTime:

        # Constructors
        DateTime() except +
        DateTime(int, int, int) except +
        DateTime(int, int, int, int, int, int) except +
        DateTime(int, int, int, int, int, int, double) except +
        DateTime(int, int, int, int, int, double) except +
        DateTime(const double) except +
        DateTime(const DateTime &) except +
        DateTime(const string &) except +
       
        # Comparison operators 
        bool operator==(const DateTime &)
        bool operator!=(const DateTime &)
        bool operator<(const DateTime &)
        bool operator<=(const DateTime &)
        bool operator>(const DateTime &)
        bool operator>=(const DateTime &)

        # Math operators
        DateTime & operator=(const DateTime &)
        DateTime & operator=(const string &)
        DateTime operator+(const TimeDelta &)
        DateTime operator+(const double & s)
        DateTime operator-(const TimeDelta & ts)
        DateTime operator-(const double & s)

        TimeDelta operator-(const DateTime & ts)

        # Get methods
        int dayOfYear()
        double secondsOfDay()
        int dayOfWeek()
        double ordinal()

        # Get and set with respect to fixed epoch
        double secondsSinceEpoch()
        double secondsSinceEpoch(const DateTime &)
        void secondsSinceEpoch(double)

        # Output methods
        string isoformat()

        # Parsing methods
        void strptime(const string &)

# end of file
