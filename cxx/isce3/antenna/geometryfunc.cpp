// Definition of the antenna-related geometry functions
#include "geometryfunc.h"

#include <isce3/core/Projections.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Vector.h>
#include <isce3/except/Error.h>
#include <isce3/geometry/geometry.h>
#include <isce3/math/RootFind1dBracket.h>

// Aliases
using namespace isce3::core;
namespace geom = isce3::geometry;
namespace ant = isce3::antenna;
using VecXd = Eigen::VectorXd;

// Helper local functions

/**
 * @internal
 * Helper function to get slant range and LLH and convergence flag
 * @param[in] el_theta : either elevation or theta angle in radians
 * depending on the "frame" object.
 * @param[in] az_phi : either azimuth or phi angle in radians depending
 * on the "frame" object.
 * @param[in] pos_ecef : antenna/spacecraft position in ECEF (m,m,m)
 * @param[in] vel_ecef : spacecraft velocity scaled by 2/wavalength in
 * ECEF (m/s,m/s,m/s)
 * @param[in] quat : isce3 quaternion object for transformation from antenna
 * body-fixed to ECEF
 * @param[in] dem_interp (optional): isce3 DEMInterpolator object
 * w.r.t ellipsoid. Default is zero height.
 * @param[in] abs_tol (optional): Abs error/tolerance in height estimation (m)
 * between desired input height and final output height. Default is 0.5.
 * @param[in] max_iter (optional): Max number of iterations in height
 * estimation. Default is 10.
 * @param[in] frame (optional): isce3 Frame object to define antenna spherical
 *  coordinate system. Default is based on "EL_AND_AZ" spherical grid.
 * @param[in] ellips (optional): isce3 Ellipsoid object defining the
 * ellipsoidal planet. Default is WGS84 ellipsoid.
 * @return a tuple of three values : slantrange (m), Doppler (Hz), a bool
 * which is true if height tolerance is met, false otherwise.
 * @exception InvalidArgument, RuntimeError
 */
static std::tuple<double, double, bool> _get_sr_dop_conv(double el_theta,
        double az_phi, const Vec3& pos_ecef, const Vec3& vel_ecef,
        const Quaternion& quat, const geom::DEMInterpolator& dem_interp,
        double abs_tol, int max_iter, const ant::Frame& frame,
        const Ellipsoid& ellips)
{
    // pointing sphercial to cartesian in Antenna frame
    auto pnt_xyz = frame.sphToCart(el_theta, az_phi);
    // pointing from Ant XYZ to global ECEF
    auto pnt_ecef = quat.rotate(pnt_xyz);
    // get slant range and target position on/above the ellipsoid
    Vec3 tg_ecef, tg_llh;
    double sr;
    auto iter_info = geom::srPosFromLookVecDem(sr, tg_ecef, tg_llh, pos_ecef,
            pnt_ecef, dem_interp, abs_tol, max_iter, ellips);
    bool convergence {true};
    if (iter_info.second > abs_tol)
        convergence = false;

    double doppler = vel_ecef.dot(pnt_ecef);

    return {sr, doppler, convergence};
}

// Antenna to Radar functions
std::tuple<double, double, bool> ant::ant2rgdop(double el_theta, double az_phi,
        const Vec3& pos_ecef, const Vec3& vel_ecef, const Quaternion& quat,
        double wavelength, const geom::DEMInterpolator& dem_interp,
        double abs_tol, int max_iter, const ant::Frame& frame,
        const Ellipsoid& ellips)
{
    if (!(wavelength > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Bad value for wavelength!");

    const auto vel_ecef_cst = (2. / wavelength) * vel_ecef;

    return _get_sr_dop_conv(el_theta, az_phi, pos_ecef, vel_ecef_cst, quat,
            dem_interp, abs_tol, max_iter, frame, ellips);
}

std::tuple<VecXd, VecXd, bool> ant::ant2rgdop(
        const Eigen::Ref<const VecXd>& el_theta, double az_phi,
        const Vec3& pos_ecef, const Vec3& vel_ecef, const Quaternion& quat,
        double wavelength, const geom::DEMInterpolator& dem_interp,
        double abs_tol, int max_iter, const ant::Frame& frame,
        const Ellipsoid& ellips)
{
    if (wavelength <= 0.0)
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Bad value for wavelength!");

    // initialization and vector allocations
    const auto vel_ecef_cst = (2. / wavelength) * vel_ecef;
    auto ang_size = el_theta.size();
    VecXd slantrange(ang_size);
    VecXd doppler(ang_size);
    bool converge {true};

    // FIXME OpenMP work sharing on this loop causes slowdown
    // for unknown reasons in conda environment
    for (decltype(ang_size) idx = 0; idx < ang_size; ++idx) {
        auto [sr, dop, flag] =
                _get_sr_dop_conv(el_theta[idx], az_phi, pos_ecef, vel_ecef_cst,
                        quat, dem_interp, abs_tol, max_iter, frame, ellips);
        slantrange(idx) = sr;
        doppler(idx) = dop;
        if (!flag)
            converge = false;
    }
    return {slantrange, doppler, converge};
}

// Antenna to Geometry
std::tuple<Vec3, bool> ant::ant2geo(double el_theta, double az_phi,
        const Vec3& pos_ecef, const Quaternion& quat,
        const geom::DEMInterpolator& dem_interp, double abs_tol, int max_iter,
        const ant::Frame& frame, const Ellipsoid& ellips)
{
    // pointing sphercial to cartesian in Antenna frame
    auto pnt_xyz = frame.sphToCart(el_theta, az_phi);
    // pointing from Ant XYZ to global ECEF
    auto pnt_ecef = quat.rotate(pnt_xyz);
    // get slant range and target position on/above the ellipsoid
    Vec3 tg_ecef, tg_llh;
    double sr;
    auto iter_info = geom::srPosFromLookVecDem(sr, tg_ecef, tg_llh, pos_ecef,
            pnt_ecef, dem_interp, abs_tol, max_iter, ellips);
    bool convergence {true};
    if (iter_info.second > abs_tol)
        convergence = false;

    return {tg_llh, convergence};
}

std::tuple<std::vector<Vec3>, bool> ant::ant2geo(
        const Eigen::Ref<const VecXd>& el_theta, double az_phi,
        const Vec3& pos_ecef, const Quaternion& quat,
        const geom::DEMInterpolator& dem_interp, double abs_tol, int max_iter,
        const ant::Frame& frame, const Ellipsoid& ellips)
{
    // initialize and allocate vectors
    auto ang_size {el_theta.size()};
    std::vector<Vec3> tg_llh_vec(ang_size);
    bool converge {true};

    // FIXME OpenMP work sharing on this loop causes slowdown
    // for unknown reasons in conda environment
    for (decltype(ang_size) idx = 0; idx < ang_size; ++idx) {
        auto [tg_llh, flag] = ant::ant2geo(el_theta(idx), az_phi, pos_ecef,
                quat, dem_interp, abs_tol, max_iter, frame, ellips);

        if (!flag)
            converge = false;
        tg_llh_vec[idx] = tg_llh;
    }
    return {tg_llh_vec, converge};
}

Vec3 ant::rangeAzToXyz(double slant_range, double az, const Vec3& pos_ecef,
        const Quaternion& quat, const geom::DEMInterpolator& dem_interp,
        double el_min, double el_max, double el_tol, const ant::Frame& frame)
{
    // Get ellipsoid associated with DEM.
    const auto ellipsoid = makeProjection(dem_interp.epsgCode())->ellipsoid();

    // EL defines a 3D position.
    const auto el2xyz =
            [&](double el) {
                const auto line_of_sight_rcs = frame.sphToCart(el, az);
                const auto line_of_sight_ecef = quat.rotate(line_of_sight_rcs);
                // As of writing, newer compilers (GCC 12.2 and clang 15.0)
                // seem to mess this up if Vec3 ctor is omitted (the return
                // value is always equal to pos_ecef).
                return Vec3(pos_ecef + slant_range * line_of_sight_ecef);
            };

    // Given a 3D position we can convert to LLH and compare to DEM.
    const auto height_error =
            [&](double el) {
                const auto target = el2xyz(el);
                const auto llh = ellipsoid.xyzToLonLat(target);
                return llh[2] - dem_interp.interpolateLonLat(llh[0], llh[1]);
            };

    double el_solution = 0.0;
    auto errcode = isce3::math::find_zero_brent(
            el_min, el_max, height_error, el_tol, &el_solution);

    if (errcode != isce3::error::ErrorCode::Success) {
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                std::string("rangeAzToXyz failed with error (") +
                        isce3::error::getErrorString(errcode) +
                        std::string(").  Current solution = ") +
                        std::to_string(el_solution));
    }
    return el2xyz(el_solution);
}
