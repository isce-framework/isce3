#include "geo2rdr.h"

void isce::geocode::geo2rdr(double x, double y,
        double & azimuthTime, double & slantRange,
        isce::geometry::DEMInterpolator & demInterp,
        isce::core::ProjectionBase * proj,
        const isce::core::Orbit& orbit,
        const isce::core::LUT2d<double>& doppler,
        const isce::core::Ellipsoid & ellipsoid,
        const double wavelength,
        const isce::core::LookSide & lookSide,
        double threshold,
        int numiter)
{
    // coordinate in the output projection system
    const isce::core::Vec3 xyz{x, y, 0.0};

    // transform the xyz in the output projection system to llh
    isce::core::Vec3 llh = proj->inverse(xyz);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // Perform geo->rdr iterations
    int geostat = isce::geometry::geo2rdr(
                    llh, ellipsoid, orbit, doppler,
                    azimuthTime, slantRange, wavelength,
                    lookSide, threshold, numiter, 1.0e-8);

    // Check convergence
    if (geostat == 0) {
        azimuthTime = std::numeric_limits<double>::quiet_NaN();
        slantRange = std::numeric_limits<double>::quiet_NaN();
        return;
    }
}
