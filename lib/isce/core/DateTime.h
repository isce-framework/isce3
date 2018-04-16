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

// DateTime declaration
struct isce::core::DateTime {

    int year;
    int months;
    int days;
    int hours;
    int minutes;
    int seconds;
    double frac;

    // Constructors
    DateTime() : DateTime(0.0) {};
    DateTime(double ord);
    DateTime(int yy, int mm, int dd);
    DateTime(int yy, int mm, int dd, int hh, int mn, int ss);
    DateTime(int yy, int mm, int dd, int hh, int mn, double ss);
    DateTime(int yy, int mm, int dd, int hh, int mn, int ss, double ff);
    DateTime(const DateTime& ts);
    DateTime(const std::string &);

    // Init function to be used by constructors
    void _init(int yy, int mm, int dd, int hh, int mn, int ss, double ff);
    void _normalize_time();
    void _normalize_date();
    void _normalize();

    // Comparison operators
    bool operator<( const DateTime &ts) const;
    bool operator>( const DateTime &ts) const;
    bool operator<=( const DateTime &ts) const;
    bool operator>=( const DateTime &ts) const;
    bool operator==( const DateTime &ts) const;
    bool operator!=( const DateTime &ts) const;

    // Math operators
    DateTime& operator=(const DateTime& ts);
    DateTime& operator+=(const TimeDelta& ts);
    DateTime& operator+=(const double& s);
    DateTime& operator-=(const TimeDelta& ts);
    DateTime& operator-=(const double& s);

    DateTime operator+(const TimeDelta& ts) const;
    DateTime operator+(const double& s) const;
    DateTime operator-(const TimeDelta& ts) const;
    DateTime operator-(const double& s) const;

    TimeDelta operator-(const DateTime& ts) const;

    // Get methods
    int dayOfYear() const;
    double secondsOfDay() const;
    int dayOfWeek() const;
    double ordinal() const;

    // Get and set with respect to fixed epoch
    double secondsSinceEpoch() const;
    void secondsSinceEpoch(double);
  
    // Output methods
    std::string isoformat() const;

    // Parsing methods
    void strptime(const std::string &);
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
