//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_DATETIME_H__
#define __ISCE_CORE_DATETIME_H__

#include <chrono>
#include <string>

namespace isce { namespace core {
    struct DateTime {
        std::chrono::time_point<std::chrono::high_resolution_clock> t;

        DateTime() : t() {}
        DateTime(const DateTime &dt) : t(dt.t) {}
        // Note that these constructors leverage the assignment operators given their relative
        // complexity
        DateTime(const std::string &dts) { *this = dts; }
        DateTime(const double dtd) { *this = dtd; }
        inline DateTime& operator=(const DateTime&);
        DateTime& operator=(const std::string&);
        DateTime& operator=(const double);

        // Wrapped boolean comparisons
        inline bool operator==(const DateTime &dt) const { return t == dt.t; }
        inline bool operator!=(const DateTime &dt) const { return t != dt.t; }
        inline bool operator<(const DateTime &dt) const { return t < dt.t; }
        inline bool operator<=(const DateTime &dt) const { return t <= dt.t; }
        inline bool operator>(const DateTime &dt) const { return t > dt.t; }
        inline bool operator>=(const DateTime &dt) const { return t >= dt.t; }

        // DateTime + duration arithmetic
        inline DateTime& operator+=(const double);
        inline DateTime& operator-=(const double);
        inline DateTime operator+(const double) const;
        inline DateTime operator-(const double) const;
        
        // DateTime - DateTime differential
        inline double operator-(const DateTime&) const;

        // Built-in conversion to time-as-double (assumes t_since_epoch)
        operator double() {
            return static_cast<std::chrono::duration<double>>(t.time_since_epoch()).count();
        }

        std::string toIsoString() const;
    };

    inline DateTime& DateTime::operator=(const DateTime &rhs) {
        t = rhs.t;
        return *this;
    }

    inline DateTime& DateTime::operator+=(const double rhs) {
        t += std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double>(rhs));
        return *this;
    }

    inline DateTime& DateTime::operator-=(const double rhs) {
        t -= std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double>(rhs));
        return *this;
    }

    inline DateTime DateTime::operator+(const double rhs) const {
        return DateTime(*this) += rhs;
    }

    inline DateTime DateTime::operator-(const double rhs) const {
        return DateTime(*this) -= rhs;
    }

    inline double DateTime::operator-(const DateTime &rhs) const {
        // Casting to duration preserves double-precision DateTime delta, count() gets the double()
        // value of a duration. This operation has a different purpose in that we want to return
        // a "duration" in seconds that is the difference between two DateTime points
        return static_cast<std::chrono::duration<double>>(t - rhs.t).count();
    }
}}

#endif
