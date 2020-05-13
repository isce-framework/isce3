#include <isce/io/forward.h>
#include <isce/geometry/DEMInterpolator.h>
#include <isce/core/Projections.h>
#include <isce/core/Orbit.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/geometry/geometry.h>
namespace isce { namespace geocode {

    void geo2rdr(double x, double y,
        double & azimuthTime, double & slantRange,
        isce::geometry::DEMInterpolator & demInterp,
        isce::core::ProjectionBase * proj,
        const isce::core::Orbit& orbit,
        const isce::core::LUT2d<double>& doppler,
        const isce::core::Ellipsoid & ellipsoid,
        const double wavelength,
        const isce::core::LookSide & lookSide,
        double threshold,
        int numiter);
}
}
