//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2017-2018

#include <cmath>
#include "TimeDelta.h"

//Normalize function
void isce::core::TimeDelta::
_normalize()
{
    //Adjust fractional part
    {
        int ipart = frac - (frac < 0);
        frac -= ipart;
        seconds += ipart;
    }

    {
        int ipart = (seconds/MIN_TO_SEC) - (seconds < 0);
        seconds -= ipart * MIN_TO_SEC;
        minutes += ipart;
    }

    {
        int ipart = (minutes/HOUR_TO_MIN) - (minutes < 0);
        minutes -= ipart * HOUR_TO_MIN;
        hours += ipart;
    }

    {
        int ipart = (hours/DAY_TO_HOUR) - (hours < 0);
        hours -= ipart * DAY_TO_HOUR;
        days += ipart;
    }

}

//Constructors
void isce::core::TimeDelta::
_init(int dd, int hh, int mm, int ss, double ff)
{
    days = dd;
    hours = hh;
    minutes = mm;
    seconds = ss;
    frac = ff;
    _normalize();
}

isce::core::TimeDelta::
TimeDelta() : TimeDelta(0.0) {}

isce::core::TimeDelta::
TimeDelta(double ss)
{
    int ipart = ss;
    double fpart = ss - ipart;
    _init(0,0,0,ipart,fpart);
}

isce::core::TimeDelta::
TimeDelta(int hh, int mm, int ss)
{
    _init(0,hh,mm,ss,0);
}

isce::core::TimeDelta::
TimeDelta(int hh, int mm, double ss)
{
    int ipart = ss;
    double fpart = ss - ipart;
    _init(0,hh,mm,ipart,fpart);
}

isce::core::TimeDelta::
TimeDelta(int hh, int mm, int ss, double ff)
{
    _init(0,hh,mm,ss,ff);
}

isce::core::TimeDelta::
TimeDelta(int dd, int hh, int mm, int ss, double ff)
{
    _init(dd,hh,mm,ss,ff);
}

isce::core::TimeDelta::
TimeDelta(const TimeDelta & ts)
{
    _init(ts.days, ts.hours, ts.minutes, ts.seconds, ts.frac);
}


//Return as double functions
double isce::core::TimeDelta::
getTotalSeconds() const
{
    return days*DAY_TO_SEC + hours * HOUR_TO_SEC + minutes * MIN_TO_SEC + seconds + frac;
}

double isce::core::TimeDelta::
getTotalMinutes() const
{
    return getTotalSeconds()/(1.0 * MIN_TO_SEC);
}

double isce::core::TimeDelta::
getTotalHours() const
{
    return getTotalSeconds()/(1.0 * HOUR_TO_SEC);
}

double isce::core::TimeDelta::
getTotalDays() const
{
    return getTotalSeconds()/ (1.0*DAY_TO_SEC);
}

//Comparison operators
bool isce::core::TimeDelta::
operator<(const TimeDelta &ts) const
{
    return *this < ts.getTotalSeconds();
}

bool isce::core::TimeDelta::
operator<(double ts) const
{
    return getTotalSeconds() < ts;
}


bool isce::core::TimeDelta::
operator>(const TimeDelta &ts) const
{
    return *this > ts.getTotalSeconds();
}

bool isce::core::TimeDelta::
operator>(double ts) const
{
    return getTotalSeconds() > ts;
}

bool isce::core::TimeDelta::
operator<=(const TimeDelta &ts) const
{
    return !(*this > ts.getTotalSeconds());
}

bool isce::core::TimeDelta::
operator<=(double ts) const
{
    return !(getTotalSeconds() > ts);
}

bool isce::core::TimeDelta::
operator>=(const TimeDelta &ts) const
{
    return !( *this < ts.getTotalSeconds());
}

bool isce::core::TimeDelta::
operator>=(double ts) const
{
    return !(getTotalSeconds() < ts);
}

bool isce::core::TimeDelta::
operator==(const TimeDelta &ts) const
{
    return (*this == ts.getTotalSeconds());
}

bool isce::core::TimeDelta::
operator==(double ts) const
{
    return getTotalSeconds() == ts;
}

bool isce::core::TimeDelta::
operator!=(const TimeDelta &ts) const
{
    return !(*this == ts.getTotalSeconds());
}

bool isce::core::TimeDelta::
operator!=(double ts) const
{
    return !(*this == ts);
}


//Math operators
isce::core::TimeDelta &
isce::core::TimeDelta::
operator=(const TimeDelta &ts) {
    _init(ts.days, ts.hours, ts.minutes, ts.seconds, ts.frac);
    _normalize();
    return *this;
}

isce::core::TimeDelta &
isce::core::TimeDelta::
operator=(double ss) {
    int ipart = ss;
    double fpart = ss - ipart;
    _init(0, 0, 0, ipart, fpart);
    return *this;
}

isce::core::TimeDelta
isce::core::TimeDelta::
operator+(const TimeDelta &ts) const
{
    return TimeDelta(days+ts.days, hours + ts.hours, 
            minutes + ts.minutes, seconds + ts.seconds,
            frac + ts.frac);
}

isce::core::TimeDelta
isce::core::TimeDelta::
operator+(const double &s) const
{
    return TimeDelta(days, hours, minutes, seconds, frac + s);
}

isce::core::TimeDelta
isce::core::TimeDelta::
operator-(const TimeDelta &ts) const
{
    return TimeDelta(days-ts.days, hours - ts.hours,
            minutes - ts.minutes, seconds - ts.seconds,
            frac - ts.frac);
}

isce::core::TimeDelta
isce::core::TimeDelta::
operator-(const double& s) const
{
    return TimeDelta(days, hours, minutes, seconds, frac - s);
}


isce::core::TimeDelta &
isce::core::TimeDelta::
operator+=(const TimeDelta& ts) 
{
    days += ts.days;
    hours += ts.hours;
    minutes += ts.minutes;
    seconds += ts.seconds;
    frac += ts.frac;
    _normalize();
    return *this;
}

isce::core::TimeDelta &
isce::core::TimeDelta::
operator+=(const double& s)
{
    frac += s;
    _normalize();
    return *this;
}

isce::core::TimeDelta &
isce::core::TimeDelta::
operator-=(const TimeDelta& ts)
{
    days -= ts.days;
    hours -= ts.hours;
    minutes -= ts.minutes;
    seconds -= ts.seconds;
    frac -= ts.frac;
    _normalize();
    return *this;
}

isce::core::TimeDelta &
isce::core::TimeDelta::
operator-=(const double& s)
{
    frac -= s;
    _normalize();
    return *this;
}

isce::core::TimeDelta
isce::core::TimeDelta::
operator*(const double& s) const
{
    return TimeDelta(getTotalSeconds()*s);
}

isce::core::TimeDelta &
isce::core::TimeDelta::
operator*=(const double& s)
{
    _init(0,0,0,0, getTotalSeconds()*s);
    return *this;
}

isce::core::TimeDelta
isce::core::TimeDelta::
operator/(const double& s) const
{
    return TimeDelta(getTotalSeconds()/s);
}

isce::core::TimeDelta &
isce::core::TimeDelta::
operator/=(const double& s)
{
    _init(0,0,0,0,getTotalSeconds()/s);
    return *this;
}

// end of file
