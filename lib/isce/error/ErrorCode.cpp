#include "ErrorCode.h"

namespace isce { namespace error {

std::string getErrorString(ErrorCode status)
{
    switch (status) {
        case ErrorCode::Success:
            return "the operation completed without errors";
        case ErrorCode::OrbitInterpSizeError:
            return "insufficient orbit state vectors to form interpolant";
        case ErrorCode::OrbitInterpDomainError:
            return "interpolation point outside orbit domain";
        case ErrorCode::OrbitInterpUnknownMethod:
            return "unexpected orbit interpolation method";
        default:
            return "unknown error code";
    }
}

}}
