#include "GeoTransform.h"

#include <isce3/except/Error.h>

namespace isce { namespace io { namespace gdal {

GeoTransform::GeoTransform(const std::array<double, 6> & coeffs)
:
    x0(coeffs[0]),
    y0(coeffs[3]),
    dx(coeffs[1]),
    dy(coeffs[5])
{
    if (coeffs[2] != 0. || coeffs[4] != 0.) {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), "unsupported geotransform orientation");
    }
}

std::array<double, 6> GeoTransform::getCoeffs() const
{
    return { x0, dx, 0., y0, 0., dy };
}

bool operator==(const GeoTransform & lhs, const GeoTransform & rhs)
{
    return lhs.x0 == rhs.x0 && lhs.y0 == rhs.y0 && lhs.dx == rhs.dx && lhs.dy == rhs.dy;
}

bool operator!=(const GeoTransform & lhs, const GeoTransform & rhs)
{
    return !(lhs == rhs);
}

}}}
