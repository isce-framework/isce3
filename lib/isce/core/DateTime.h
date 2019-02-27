//-*- C++ -*-
//-*- coding: utf-8 -*-
//
//
// Author: Piyush Agram
// Copyright 2017-2018

#ifndef ISCE_CORE_DATETIME_H
#define ISCE_CORE_DATETIME_H

#include <cassert>
#include <cmath>
#include <string>
#include "TimeDelta.h"

// Declaration
namespace isce {
    namespace core {
        struct DateTime;
    }
}

/** Data structure to store date time to nano-sec precision*/
struct isce::core::DateTime {

    int year;
    int months;
    int days;
    int hours;
    int minutes;
    int seconds;
    double frac;

    /**Default constructor
     *
     * Initialize to origin of GPS time 1970-01-01T00:00:00.0*/
    DateTime() : DateTime(1970, 1, 1) {};

    /**Constructor using ordinal*/
    DateTime(double ord);

    /**Constructor using year, month, day of month*/
    DateTime(int yy, int mm, int dd);

    /**Constructor using date, hours, mins and integer seconds*/
    DateTime(int yy, int mm, int dd, int hh, int mn, int ss);

    /**Constructor using date, hours, mins, and floating point seconds */
    DateTime(int yy, int mm, int dd, int hh, int mn, double ss);

    /**Constructor using date, hours, mins, integer secs and fractional secs */
    DateTime(int yy, int mm, int dd, int hh, int mn, int ss, double ff);

    /**Copy constructor*/
    DateTime(const DateTime& ts);

    /**Construct from a string representation in ISO-8601 format*/
    DateTime(const std::string &);

    ///@cond
    void _init(int yy, int mm, int dd, int hh, int mn, int ss, double ff);
    void _normalize_time();
    void _normalize_date();
    void _normalize();
    ///@endcond

    // Comparison operators
    bool operator<( const DateTime &ts) const;
    bool operator>( const DateTime &ts) const;
    bool operator<=( const DateTime &ts) const;
    bool operator>=( const DateTime &ts) const;
    bool operator==( const DateTime &ts) const;
    bool operator!=( const DateTime &ts) const;

    // Math operators
    DateTime& operator=(const DateTime& ts);
    DateTime& operator=(const std::string &);
    DateTime& operator+=(const TimeDelta& ts);
    DateTime& operator+=(const double& s);
    DateTime& operator-=(const TimeDelta& ts);
    DateTime& operator-=(const double& s);

    DateTime operator+(const TimeDelta& ts) const;
    DateTime operator+(const double& s) const;
    DateTime operator-(const TimeDelta& ts) const;
    DateTime operator-(const double& s) const;

    TimeDelta operator-(const DateTime& ts) const;

    /** Return day of year*/
    int dayOfYear() const;

    /**Return seconds of day*/
    double secondsOfDay() const;

    /**Return day of week*/
    int dayOfWeek() const;

    /**Return ordinal - time since GPS time origin*/
    double ordinal() const;

    /**Return ordinal - time since GPS time origin*/
    double secondsSinceEpoch() const;

    /**Return time elapsed since given DateTime epoch*/
    double secondsSinceEpoch(const DateTime &) const;

    /**Return time elapsed since given ordinal epoch*/
    void secondsSinceEpoch(double);
  
    /**Return date formatted as ISO-8601 string*/
    std::string isoformat() const;

    /**Parse a given string in ISO-8601 format*/
    void strptime(const std::string &, const std::string & sep = "T");
};

// Some constants
namespace isce {
    namespace core {

        // Constants for default constructors
        const DateTime MIN_DATE_TIME = DateTime(1970, 1, 1);
        const std::string UNINITIALIZED_STRING = "uninitialized";

        static const int DaysInMonths[] = {31,28,31,
                                       30,31,30,
                                       31,31,30,
                                       31,30,31};

        static const int DaysBeforeMonths[] = {0,31,59,
                                              90,120,151,
                                              181,212,243,
                                              273,304,334};
        static const int DAY_TO_YEAR = 365;
        static const int DAYSPER100  = 36524;
        static const int DAYSPER400  = 146097;
        static const int DAYSPER4    = 1461;
        static const int MAXORDINAL  = 3652059;
        static const double TOL_SECONDS = 1.0e-11;

        // Handful of utility functions
        bool _is_leap(int);
        int _days_in_month(int, int);
        int _days_before_year(int);
        int _days_before_month(int, int);
        int _ymd_to_ord(int, int, int);
        void _ord_to_ymd(int, int &, int &, int &);
    }
}

#endif

// end if
