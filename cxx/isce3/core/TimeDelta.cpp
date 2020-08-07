//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2017-2018

#include "TimeDelta.h"

#include <cmath>
#include <cstdint>
#include <isce3/except/Error.h>
#include <limits>

using isce3::except::DomainError;

//Normalize function
void isce3::core::TimeDelta::_normalize()
{
    // Promote fields to avoid intermediate overflows.
    using T = std::int64_t;
    T d{days}, h{hours}, m{minutes}, s{seconds};
    // Fail if seconds doesn't fit in an integer:
    // int32: 2**31 s ~ 68 years
    // int64: 2**63 s ~ 10**11 years
    // Just consider the bounding case where fraction & seconds have same sign.
    double maxsec = std::abs(frac) + std::fabs(s);
    if (maxsec >= static_cast<double>(std::numeric_limits<T>::max())) {
        throw DomainError(ISCE_SRCINFO(), "Time interval too large (seconds).");
    }
    //Adjust fractional part
    {
        T ipart = static_cast<T>(frac - (frac < 0));
        frac -= static_cast<double>(ipart);
        s += ipart;
    }

    {
        T ipart = (s / MIN_TO_SEC) - (s < 0);
        s -= ipart * MIN_TO_SEC;
        m += ipart;
    }

    {
        T ipart = (m / HOUR_TO_MIN) - (m < 0);
        m -= ipart * HOUR_TO_MIN;
        h += ipart;
    }

    {
        T ipart = (h / DAY_TO_HOUR) - (h < 0);
        h -= ipart * DAY_TO_HOUR;
        d += ipart;
    }
    // At this point only days might overflow.
    if (d > std::numeric_limits<decltype(days)>::max()) {
        throw DomainError(ISCE_SRCINFO(), "Time interval too large (days).");
    }
    // Truncate to storage type.
    days = static_cast<decltype(days)>(d);
    hours = static_cast<decltype(hours)>(h);
    minutes = static_cast<decltype(minutes)>(m);
    seconds = static_cast<decltype(seconds)>(s);
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
TimeDelta(double ss)
{
    _init(0,0,0,0,ss);
}

isce3::core::TimeDelta::
TimeDelta(int hh, int mm, int ss)
{
    _init(0,hh,mm,ss,0);
}

isce3::core::TimeDelta::
TimeDelta(int hh, int mm, double ss)
{
    _init(0,hh,mm,0,ss);
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
    // Careful to avoid intermediate overflow. Biggest multiplier
    // log2(DAY_TO_SEC) is roughly 16 bits, and four adds means a couple extra
    // bits could be needed. Assuming int is 32 bits, promoting to 64 bits will
    // be plenty even when `this` is not normalized.
    //
    // Conversion of the integer seconds sum to double will lose data when
    // greater than 2**52 seconds (100 million years)--not too bad.
    //
    // Underflow of frac is a bigger problem. You lose nanosecond precision
    // after a few weeks. Of course, all these caveats are why we have a
    // TimeDelta and don't just use double everywhere.
    using T = std::int64_t;
    return frac +
           static_cast<double>(days * T(DAY_TO_SEC) + hours * T(HOUR_TO_SEC) +
                               minutes * T(MIN_TO_SEC) + T(seconds));
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
    _init(0, 0, 0, 0, ss);
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
