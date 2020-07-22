#pragma once

#include <stdexcept>
#include <string>

namespace isce3 {
    //! The isce3::except namespace
    namespace except {

    struct SrcInfo {
        const char* file;
        const int line;
        const char* func;
    };

// macro-expanded pseudo-constructors
#define ISCE_SRCINFO() { __FILE__, __LINE__, __PRETTY_FUNCTION__ }
#define ISCE_ERROR(T, str) isce3::except::Error<T>(ISCE_SRCINFO(), str)

    template<typename T>
    class Error : public T {
        public:
            const SrcInfo info;

        Error(const SrcInfo& info);
        Error(const SrcInfo& info, std::string msg);
    };

    // STL exception types
    using DomainError = Error<std::domain_error>;
    using InvalidArgument = Error<std::invalid_argument>;
    using LengthError = Error<std::length_error>;
    using OutOfRange = Error<std::out_of_range>;
    using OverflowError = Error<std::overflow_error>;
    using RuntimeError = Error<std::runtime_error>;

    // special exception type for errors returned from GDAL API functions
    using GDALError = Error<std::runtime_error>;
}}
