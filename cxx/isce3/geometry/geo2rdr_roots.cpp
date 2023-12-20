#include "geo2rdr_roots.h"

#include <isce3/core/LUT2d.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Vector.h>
#include <isce3/except/Error.h>

#include "detail/Geo2Rdr.h"

using namespace isce3::core;
using isce3::error::ErrorCode;

namespace isce3 { namespace geometry {

int geo2rdr_bracket(const Vec3& x, const Orbit& orbit,
        const LUT2d<double>& doppler, double& aztime, double& range,
        double wavelength, LookSide side, double tolAzTime,
        std::optional<double> timeStart, std::optional<double> timeEnd)
{
    const ErrorCode err = isce3::geometry::detail::geo2rdr_bracket(
            &aztime, &range, x, orbit, doppler, wavelength, side,
            {tolAzTime, timeStart, timeEnd});
    return err == ErrorCode::Success;
}

}} // namespace isce3::geometry
