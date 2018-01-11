//
// Author: Joshua Cohen
// Copyright 2017
//
// AUTHOR'S NOTE: Current implementations of string parsing (copy/= from string and toIsoString to
//                string) use the put_time/get_time methods defined in <iomanip> from the C++11
//                standard, however the function wasn't actually added until GCC 5. Therefore to
//                allow for building at the moment, these functions are protected from being
//                defined (both here and in the .cpp implementation) by version checking the
//                preprocessor macros (similar to the issue with std::isnan between gcc and clang).
//                These guards will be replaced with internal guards in the functions themselves
//                that will have implementations for GCCs earlier than GCC 5.

#ifndef __ISCE_CORE_DATETIME_H__
#define __ISCE_CORE_DATETIME_H__

#include <chrono>
#include <string>

namespace isce { namespace core {
    struct DateTime {
        std::chrono::time_point<std::chrono::high_resolution_clock> t;

        DateTime() : t() {}
        DateTime(const DateTime &dt) : t(dt.t) {}
        //#if __cplusplus >= 201103L
        #if 0
        DateTime(const std::string &dts) { *this = dts; }
        #endif
        // Note that these constructors leverage the assignment operators given their relative
        // complexity
        DateTime(double dtd) { *this = dtd; }
        inline DateTime& operator=(const DateTime&);
        //#if __cplusplus >= 201103L
        #if 0
        DateTime& operator=(const std::string&);
        #endif
        DateTime& operator=(double);

        // Wrapped boolean comparisons
        inline bool operator==(const DateTime &dt) const { return t == dt.t; }
        inline bool operator!=(const DateTime &dt) const { return t != dt.t; }
        inline bool operator<(const DateTime &dt) const { return t < dt.t; }
        inline bool operator<=(const DateTime &dt) const { return t <= dt.t; }
        inline bool operator>(const DateTime &dt) const { return t > dt.t; }
        inline bool operator>=(const DateTime &dt) const { return t >= dt.t; }

        // DateTime + duration arithmetic
        inline DateTime& operator+=(double);
        inline DateTime& operator-=(double);
        inline DateTime operator+(double) const;
        inline DateTime operator-(double) const;
        
        // DateTime - DateTime differential
        inline double operator-(const DateTime&) const;

        // Built-in conversion to time-as-double (assumes t_since_epoch)
        operator double() {
            return static_cast<std::chrono::duration<double>>(t.time_since_epoch()).count();
        }

        //#if __cplusplus >= 201103L
        #if 0
        std::string toIsoString() const;
        #endif
    };

    inline DateTime& DateTime::operator=(const DateTime &rhs) {
        t = rhs.t;
        return *this;
    }

    inline DateTime& DateTime::operator+=(double rhs) {
        t += std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double>(rhs));
        return *this;
    }

    inline DateTime& DateTime::operator-=(double rhs) {
        t -= std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double>(rhs));
        return *this;
    }

    inline DateTime DateTime::operator+(double rhs) const {
        return DateTime(*this) += rhs;
    }

    inline DateTime DateTime::operator-(double rhs) const {
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
