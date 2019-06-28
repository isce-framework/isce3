//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2017-2018

#ifndef ISCE_CORE_TIMEDELTA_H
#define ISCE_CORE_TIMEDELTA_H
#pragma once

#include "forward.h"

/** Data structure to store TimeDelta to double precision seconds
 *
 * The intent of the class is to assist in translating DateTime tags
 * to double precision floats w.r.t Reference epoch for numerical
 * computation and vice-versa*/
struct isce::core::TimeDelta {

    /** Integer days */
    int days;
    /** Integer hours */
    int hours;
    /** Integer minutes */
    int minutes;
    /** Integer seconds */
    int seconds;
    /** Double precision fractional seconds */
    double frac;

    /** Empty constructor*/
    TimeDelta();

    /** Constructor with seconds */
    TimeDelta(double ss);

    /** Constructor with hours, minutes and seconds */
    TimeDelta(int hh, int mm, int ss);

    /** Constructor with hours, minutes and seconds */
    TimeDelta(int hh, int mm, double ss);

    /** Constructor with hours, minutes, seconds and fractional seconds */
    TimeDelta(int hh, int mm, int ss, double ff);

    /** Constructor with days, hours, minutes, seconds and fractional seconds*/
    TimeDelta(int days, int hours, int minutes, int seconds, double frac);

    /** Copy constructor*/
    TimeDelta(const TimeDelta& ts);

    /** Internal function for use with constructors */
    void _init(int days, int hours, int minutes, int seconds, double frac);
    /** Internal function*/
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
    TimeDelta& operator=(double ss);
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

    /** Return equivalent double precision days */
    double getTotalDays() const;
    /** Return equivalent double precision hours */
    double getTotalHours() const;
    /** Return equivalent double precision minutes */
    double getTotalMinutes() const;
    /** Return equivalent double precision seconds */
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
