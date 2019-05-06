#pragma once

#include <stdexcept>
#include <string>

namespace isce { 
    //! The isce::except namespace
    namespace except {

    struct SrcInfo {
        const char* file;
        const int line;
        const char* func;
    };

// macro-expanded pseudo-constructors
#define ISCE_SRCINFO() { __FILE__, __LINE__, __PRETTY_FUNCTION__ }
#define ISCE_ERROR(T, str) isce::except::Error<T>(ISCE_SRCINFO(), str)

    template<typename T>
    struct Error : T {
        const SrcInfo info;

        Error(const SrcInfo& info);
        Error(const SrcInfo& info, std::string msg);
    };

    using LengthError  = Error<std::length_error>;
    using RuntimeError = Error<std::runtime_error>;
}}
