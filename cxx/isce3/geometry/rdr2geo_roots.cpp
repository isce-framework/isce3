#include "rdr2geo_roots.h"

#include <isce3/core/Projections.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

using namespace isce3::core;
using isce3::error::ErrorCode;
namespace detail = isce3::geometry::detail;

namespace isce3 { namespace geometry {

int rdr2geo_bracket(double aztime, double slantRange, double doppler,
        const Orbit& orbit, const DEMInterpolator& dem, Vec3& targetXYZ,
        double wavelength, LookSide side, double tolHeight, double lookMin,
        double lookMax)
{
    // Get ellipsoid associated with DEM.
    const auto epsg = dem.epsgCode();
    const Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();
    const detail::Rdr2GeoBracketParams params{tolHeight, lookMin, lookMax};
    const ErrorCode err = detail::rdr2geo_bracket(&targetXYZ, aztime,
            slantRange, doppler, orbit, dem, ellipsoid, wavelength, side,
            params);
    return err == ErrorCode::Success;
}

}} // namespace isce3::geometry
