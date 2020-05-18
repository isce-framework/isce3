#include <isce/io/forward.h>
#include <isce/geometry/DEMInterpolator.h>
#include <isce/core/Projections.h>
#include <isce/core/Orbit.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/geometry/geometry.h>
namespace isce { namespace geocode {

    /**computes radar coordinates from for given geo coordinates, DEM, Doppler and orbit information
     * @param[in] x x coordinate in the coordinates system specified by proj input argument
     * @param[in] y y coordinate in the coordinates system specified by proj input argument
     * @param[out] azimuthTime computed azimuth time
     * @param[out] slantRange  computed slant range
     * @param[in] demInterp    DEM interpolator
     * @param[in] proj         projection which specifies the projection system in which the geo-coordinates are given
     * @param[in] orbit        orbit object
     * @param[in] doppler      Doppler LUT 
     * @param[in] ellipsoid    ellipsoid object
     * @param[in] wavelength   wavelength of the radar
     * @param[in] lookSide     look direction of the radar
     * @param[in] threshold    threshold for geo2rdr convergence
     * @param[in] numiter      number of iterations to converge in Geo2rdr
     */
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
