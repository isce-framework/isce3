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
        case ErrorCode::FailedToConverge:
            return "optimization routine failed to converge within the maximum "
                   "number of iterations";
        case ErrorCode::WrongLookSide:
            return "wrong look side";
        default:
            return "unknown error code";
    }
}

}} // namespace isce::error
