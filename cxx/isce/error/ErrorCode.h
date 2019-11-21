#pragma once

#include <string>

namespace isce { namespace error {

/** Return code type indicating the exit status of an ISCE operation */
enum class ErrorCode {
    Success = 0,
    OrbitInterpSizeError,
    OrbitInterpDomainError,
    OrbitInterpUnknownMethod,
};

/** Return a string describing the error code */
std::string getErrorString(ErrorCode);

}}
