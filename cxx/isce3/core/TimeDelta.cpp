//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2017-2018

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <isce3/except/Error.h>
#include "TimeDelta.h"

using isce3::except::DomainError;

//Normalize function
template <typename T>
static void _normalize(T& days, T& hours, T& minutes, T& seconds, double& frac)
{
    // Written assuming truncation on assignment to ipart.
    // T=double would be silently wrong.
    static_assert(std::is_integral<T>::value, "expected integer type");
    // Likewise T=unsigned would be silently wrong.
    static_assert(std::is_signed<T>::value, "expected signed integer type");
    // Fail if seconds doesn't fit in an integer:
    // int32: 2**31 s ~ 68 years
    // int64: 2**63 s ~ 10**11 years
    // Just consider the bounding case where fraction & seconds have same sign.
    if (std::abs(frac) + std::abs(seconds) >= std::numeric_limits<T>::max()) {
        throw DomainError(ISCE_SRCINFO(), "Time interval too large (seconds).");
    }
    using namespace isce3::core;
    //Adjust fractional part
    {
        T ipart = frac - (frac < 0);
        frac -= ipart;
        seconds += ipart;
    }

    {
        T ipart = (seconds/MIN_TO_SEC) - (seconds < 0);
        seconds -= ipart * MIN_TO_SEC;
        minutes += ipart;
    }

    {
        T ipart = (minutes/HOUR_TO_MIN) - (minutes < 0);
        minutes -= ipart * HOUR_TO_MIN;
        hours += ipart;
    }

    {
        T ipart = (hours/DAY_TO_HOUR) - (hours < 0);
        hours -= ipart * DAY_TO_HOUR;
        days += ipart;
    }

}

void isce3::core::TimeDelta::_normalize()
{
    ::_normalize(days, hours, minutes, seconds, frac);
}

//Constructors
void isce3::core::TimeDelta::
_init(int dd, int hh, int mm, int ss, double ff)
{
    days = dd;
    hours = hh;
    minutes = mm;
    seconds = ss;
    frac = ff;
    _normalize();
}

isce3::core::TimeDelta::
TimeDelta() : TimeDelta(0.0) {}

isce3::core::TimeDelta::
TimeDelta(const double seconds)
{
    // Careful not to overflow intermediate fields.
    std::int64_t d{0}, h{0}, m{0}, s{0};
    double frac = seconds;
    ::_normalize(d, h, m, s, frac);
    // At this point only days might overflow.
    if (d > std::numeric_limits<int>::max()) {
        throw DomainError(ISCE_SRCINFO(), "Time interval too large (days).");
    }
    // Truncate to int.
    _init(d, h, m, s, frac);
}

isce3::core::TimeDelta::
TimeDelta(int hh, int mm, int ss)
{
    _init(0,hh,mm,ss,0);
}

isce3::core::TimeDelta::
TimeDelta(int hh, int mm, double ss)
{
    int ipart = ss;
    double fpart = ss - ipart;
    _init(0,hh,mm,ipart,fpart);
}

isce3::core::TimeDelta::
TimeDelta(int hh, int mm, int ss, double ff)
{
    _init(0,hh,mm,ss,ff);
}

isce3::core::TimeDelta::
TimeDelta(int dd, int hh, int mm, int ss, double ff)
{
    _init(dd,hh,mm,ss,ff);
}

isce3::core::TimeDelta::
TimeDelta(const TimeDelta & ts)
{
    _init(ts.days, ts.hours, ts.minutes, ts.seconds, ts.frac);
}


//Return as double functions
double isce3::core::TimeDelta::
getTotalSeconds() const
{
    // Careful to avoid intermediate overflow.
    using T = std::int64_t;
    return days * T(DAY_TO_SEC)
        + hours * T(HOUR_TO_SEC)
        + minutes * T(MIN_TO_SEC)
        + T(seconds) + frac;
}

double isce3::core::TimeDelta::
getTotalMinutes() const
{
    return getTotalSeconds()/(1.0 * MIN_TO_SEC);
}

double isce3::core::TimeDelta::
getTotalHours() const
{
    return getTotalSeconds()/(1.0 * HOUR_TO_SEC);
}

double isce3::core::TimeDelta::
getTotalDays() const
{
    return getTotalSeconds()/ (1.0*DAY_TO_SEC);
}

//Comparison operators
bool isce3::core::TimeDelta::
operator<(const TimeDelta &ts) const
{
    return *this < ts.getTotalSeconds();
}

bool isce3::core::TimeDelta::
operator<(double ts) const
{
    return getTotalSeconds() < ts;
}


bool isce3::core::TimeDelta::
operator>(const TimeDelta &ts) const
{
    return *this > ts.getTotalSeconds();
}

bool isce3::core::TimeDelta::
operator>(double ts) const
{
    return getTotalSeconds() > ts;
}

bool isce3::core::TimeDelta::
operator<=(const TimeDelta &ts) const
{
    return !(*this > ts.getTotalSeconds());
}

bool isce3::core::TimeDelta::
operator<=(double ts) const
{
    return !(getTotalSeconds() > ts);
}

bool isce3::core::TimeDelta::
operator>=(const TimeDelta &ts) const
{
    return !( *this < ts.getTotalSeconds());
}

bool isce3::core::TimeDelta::
operator>=(double ts) const
{
    return !(getTotalSeconds() < ts);
}

bool isce3::core::TimeDelta::
operator==(const TimeDelta &ts) const
{
    return (*this == ts.getTotalSeconds());
}

bool isce3::core::TimeDelta::
operator==(double ts) const
{
    return getTotalSeconds() == ts;
}

bool isce3::core::TimeDelta::
operator!=(const TimeDelta &ts) const
{
    return !(*this == ts.getTotalSeconds());
}

bool isce3::core::TimeDelta::
operator!=(double ts) const
{
    return !(*this == ts);
}


//Math operators
isce3::core::TimeDelta &
isce3::core::TimeDelta::
operator=(const TimeDelta &ts) {
    _init(ts.days, ts.hours, ts.minutes, ts.seconds, ts.frac);
    _normalize();
    return *this;
}

isce3::core::TimeDelta &
isce3::core::TimeDelta::
operator=(double ss) {
    int ipart = ss;
    double fpart = ss - ipart;
    _init(0, 0, 0, ipart, fpart);
    return *this;
}

isce3::core::TimeDelta
isce3::core::TimeDelta::
operator+(const TimeDelta &ts) const
{
    return TimeDelta(days+ts.days, hours + ts.hours,
            minutes + ts.minutes, seconds + ts.seconds,
            frac + ts.frac);
}

isce3::core::TimeDelta
isce3::core::TimeDelta::
operator+(const double &s) const
{
    return TimeDelta(days, hours, minutes, seconds, frac + s);
}

isce3::core::TimeDelta
isce3::core::TimeDelta::
operator-(const TimeDelta &ts) const
{
    return TimeDelta(days-ts.days, hours - ts.hours,
            minutes - ts.minutes, seconds - ts.seconds,
            frac - ts.frac);
}

isce3::core::TimeDelta
isce3::core::TimeDelta::
operator-(const double& s) const
{
    return TimeDelta(days, hours, minutes, seconds, frac - s);
}


isce3::core::TimeDelta &
isce3::core::TimeDelta::
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

isce3::core::TimeDelta &
isce3::core::TimeDelta::
operator+=(const double& s)
{
    frac += s;
    _normalize();
    return *this;
}

isce3::core::TimeDelta &
isce3::core::TimeDelta::
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

isce3::core::TimeDelta &
isce3::core::TimeDelta::
operator-=(const double& s)
{
    frac -= s;
    _normalize();
    return *this;
}

isce3::core::TimeDelta
isce3::core::TimeDelta::
operator*(const double& s) const
{
    return TimeDelta(getTotalSeconds()*s);
}

isce3::core::TimeDelta &
isce3::core::TimeDelta::
operator*=(const double& s)
{
    _init(0,0,0,0, getTotalSeconds()*s);
    return *this;
}

isce3::core::TimeDelta
isce3::core::TimeDelta::
operator/(const double& s) const
{
    return TimeDelta(getTotalSeconds()/s);
}

isce3::core::TimeDelta &
isce3::core::TimeDelta::
operator/=(const double& s)
{
    _init(0,0,0,0,getTotalSeconds()/s);
    return *this;
}

namespace isce3 { namespace core {

TimeDelta operator*(double lhs, const TimeDelta & rhs)
{
    return TimeDelta(lhs * rhs.getTotalSeconds());
}

}}

// end of file
