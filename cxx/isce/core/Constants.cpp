#include "Constants.h"
#include <algorithm>
#include <isce/except/Error.h>
#include <map>

using isce::except::InvalidArgument;

namespace isce { namespace core {

dataInterpMethod
parseDataInterpMethod(const std::string & method)
{
    std::string m = method;
    std::transform(m.begin(), m.end(), m.begin(),
        [](unsigned char c) { return std::tolower(c); });

    std::map<std::string, dataInterpMethod> methods {
        {"sinc",      SINC_METHOD},
        {"bilinear",  BILINEAR_METHOD},
        {"bicubic",   BICUBIC_METHOD},
        {"nearest",   NEAREST_METHOD},
        {"biquintic", BIQUINTIC_METHOD}
    };
    auto it = methods.find(m);
    if (it == methods.end())
        throw InvalidArgument(ISCE_SRCINFO(), "Unknown interp method");

    return it->second;
}

}} // isce::core
