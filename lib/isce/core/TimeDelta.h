//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2017-2018

#ifndef ISCE_CORE_TIMEDELTA_H
#define ISCE_CORE_TIMEDELTA_H

#include <cstdint>

// Declaration
namespace isce {
    namespace core {
        struct TimeDelta;
    }
}

// TimeDelta declaration
struct isce::core::TimeDelta {

    // Data members
    int days;
    int hours;
    int minutes;
    int seconds;
    double frac;

    // Constructors
    TimeDelta();
    TimeDelta(double ss);
    TimeDelta(int hh, int mm, int ss);
    TimeDelta(int hh, int mm, double ss);
    TimeDelta(int hh, int mm, int ss, double ff);
    TimeDelta(int days, int hours, int minutes, int seconds, double frac);
    TimeDelta(const TimeDelta& ts);

    // Init function to be used by constructors
    void _init(int days, int hours, int minutes, int seconds, double frac);
    void _normalize();

    // Comparison operators
    bool operator<( const TimeDelta &ts) const;
    bool operator<( double ts) const;
    bool operator>( const TimeDelta &ts) const;
    bool operator>( double ts) const;
    bool operator<=( const TimeDelta &ts) const;
    bool operator<=( double ts) const;
    bool operator>=( const TimeDelta &ts) const;
    bool operator>=( double ts) const;
    bool operator==( const TimeDelta &ts) const;
    bool operator==( double ts) const;
    bool operator!=( const TimeDelta &ts) const;
    bool operator!=( double ts) const;

    // Math operators
    TimeDelta& operator=(const TimeDelta& ts);
    TimeDelta& operator+=(const TimeDelta& ts);
    TimeDelta& operator+=(const double& s);
    TimeDelta& operator-=(const TimeDelta& ts);
    TimeDelta& operator-=(const double& s);

    TimeDelta operator+(const TimeDelta& ts) const;
    TimeDelta operator+(const double& s) const;
    TimeDelta operator-(const TimeDelta& ts) const;
    TimeDelta operator-(const double& s) const;

    TimeDelta operator*(const double& s) const;
    TimeDelta& operator*=(const double& s);

    TimeDelta operator/(const double& s) const;
    TimeDelta& operator/=(const double& s);

    // Get methods
    double getTotalDays() const;
    double getTotalHours() const;
    double getTotalMinutes() const;
    double getTotalSeconds() const;
};

// Some constants
namespace isce {
    namespace core {
        const int MIN_TO_SEC=60;
        const int HOUR_TO_SEC=3600;
        const int HOUR_TO_MIN=60;
        const int DAY_TO_SEC=86400;
        const int DAY_TO_MIN=1440;
        const int DAY_TO_HOUR=24;
    }
}

#endif

// end of file
