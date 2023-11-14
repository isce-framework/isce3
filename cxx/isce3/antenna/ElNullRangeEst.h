#pragma once

#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <isce3/core/Attitude.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/EMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly1d.h>
#include <isce3/geometry/DEMInterpolator.h>

#include "Frame.h"

namespace isce3 { namespace antenna {

/** EL null product  */
struct NullProduct {
    /** Slant range of the null location in (m) */
    double slant_range;
    /** Elevation angle of the null location in (rad) */
    double el_angle;
    /** Doppler at the null location in (Hz) */
    double doppler;
    /** Relative magnitude of the null w.r.t left/right peaks in (linear) */
    double magnitude;
};

/**
 * A set of flags indicating convergence of iterative operations
 * used in EL null product formation
 */
struct NullConvergenceFlags {
    /** Indicates convergence of the 1-D Newton root solver */
    bool newton_solver;
    /** Indicates geometry-related convergence for echo null estimation */
    bool geometry_echo;
    /** Indicates geometry-related convergence for antenna null estimation */
    bool geometry_antenna;
};

/** EL Null Power Patterns for both echo and antenna as a function of EL angles
 */
struct NullPowPatterns {
    /** 1-D antenna null power pattern (linear) in EL formed from a pair of
     * adjacent beams */
    Eigen::ArrayXd ant;
    /** 1-D echo null power pattern (linear) in EL formed from a pair of
     * adjacent channels */
    Eigen::ArrayXd echo;
    /** Elevation (EL) angles (radians) */
    Eigen::ArrayXd el;
};

/**
 * A class for forming Null power patterns in EL direction from both
 * a pair of adjacent El-cut antenna patterns as well as the respective
 * raw echoes of two adjacent RX channels.
 * The location of null in both antenna and echo domain will be estimated
 * and their respective values in EL angle, slant range, and Doppler will
 * be reported at a specific azimuth time in orbit.
 * See the following citation and its references for algorithm, simulation and
 * analyses,
 * @cite NullPatternElPointDoc
 */
class ElNullRangeEst {
    // aliases
public:
    using RowMatrixXcf = isce3::core::EMatrix2D<std::complex<float>>;

protected:
    using Linspace_t = isce3::core::Linspace<double>;
    using tuple_null = std::tuple<isce3::core::DateTime, NullProduct,
            NullProduct, NullConvergenceFlags, NullPowPatterns>;

public:
    // constructors
    /**
     * A constructor
     * @param[in] wavelength  wavelength in (m)
     * @param[in] sr_spacing slant range spacing in (m)
     * @param[in] chirp_rate chirp rate in (Hz/sec)
     * @param[in] chirp_dur chirp duration in (sec)
     * @param[in] orbit isce3 Orbit object
     * @param[in] attitude isce3 Attitude object for quaternions
     * from antenna to ECEF
     * @param[in] dem_interp (optional) isce3 DEMInterpolator object.
     * Default is 0.0 (m) or reference ellipsoid.
     * @param[in] ant_frame (optional) isce3 antenna Frame object.
     * Default is "EL_AND_AZ" antenna coordinate system.
     * @param[in] ellips (optional) isce3 Ellipsoid object. Default is
     * WGS84 ellipsoid.
     * @param[in] el_res (optional) max EL angle resolution in (rad) used
     * for all EL angles. Default is 8.726646259971648e-06 (rad) equivalent to
     * 0.5 (mdeg). Note that if antenna patterns have finer EL angle step then
     * this value will be ignored!
     * @param[in] abs_tol_dem (optional) absolute DEM height tolerance in (m).
     * Default is 1. This is used in a recursive process of transformation
     * from antenna angle to slant range in the presence of topography/DEM data.
     * @param[in] max_iter_dem (optional) max number of iteration in meeting
     * above DEM height tolerance. Default is 20.
     * @param[in] polyfit_deg (optional) degree of polyfit used for smoothing
     * of echo null power pattern and locating its null/min location. Default
     * is 6. The even orders equal or larger than 2 is required.
     * @exception InvalidArgument
     */
    ElNullRangeEst(double wavelength, double sr_spacing, double chirp_rate,
            double chirp_dur, const isce3::core::Orbit& orbit,
            const isce3::core::Attitude& attitude,
            const isce3::geometry::DEMInterpolator& dem_interp = {},
            const Frame& ant_frame = {},
            const isce3::core::Ellipsoid& ellips = {},
            double el_res = 8.726646259971648e-06, double abs_tol_dem = 1.0,
            int max_iter_dem = 20, int polyfit_deg = 6);

    // methods
    /**
     * Generate null products from echo (measured) and antenna
     * (nominal/expected). The null product consists of azimuth time tag, null
     * relative magnitude and its location in EL and slant range, plus its
     * Doppler value given azimuth (antenna geometry)/squint(Radar geometry)
     * angle.
     * @param[in] echo_left is complex 2-D array of raw echo samples (pulse by
     * range) for the left RX channel corresponding to the left beam.
     * @param[in] echo_right is complex 2-D array of raw echo samples (pulse by
     * range) for the right RX channel corresponding to the right beam. Must
     * have the same shape as of that of left one!
     * @param[in] el_cut_left is complex array of uniformly-sampled relative or
     * absolute EL-cut antenna pattern on the left side.
     * @param[in] el_cut_right is complex array of uniformly-sampled relative or
     * absolute EL-cut antenna pattern on the right side. It must have the same
     * size as left one!
     * @param[in] sr_start is start slant range (m) for both uniformly-sampled
     * echoes in range.
     * @param[in] el_ang_start is start elevation angle for left/right EL
     * patterns in (rad)
     * @param[in] el_ang_step is step elevation angle for left/right EL patterns
     * in (rad)
     * @param[in] az_ang azimuth angle in antenna frame (similar to squint angle
     * in radar geometry) in (rad).
     * This angle determines the final Doppler centroid on top of slant range
     * value for both echo and antenna nulls.
     * @param[in] az_time (optional) azimuth time of echoes in (sec) w.r.t
     * reference epoch of orbit. If not specified, the mid azimuth time of orbit
     * will be used instead.
     * @return isce3 DateTime object representing the azimuth time tag of the
     * null.
     * @return isce3::antenna::NullProduct with members: slant_range (m),
     * el_angle (rad), doppler (Hz), and magnitude(linear) for
     * measured/estimated null product from a pair of raw echoes.
     * @return isce3::antenna::NullProduct with members: slant_range (m),
     * el_angle (rad), doppler (Hz), and magnitude(linear) for expected/nomial
     * null product from a pair of EL antenna patterns.
     * @return isce3::antenna::NullConvergenceFlags with members: newton-solver,
     * geometry_echo and geometry_antenna related to ant2rgdop convergence flag.
     * @return isce3::antenna::NullPowPatterns with members: ant, echo, and el
     * representing null power patterns (linear) for both antenna and echo as
     * a function of EL angles (radians).
     * @exception InvalidArgument, RuntimeError
     */
    std::tuple<isce3::core::DateTime, NullProduct, NullProduct,
            NullConvergenceFlags, NullPowPatterns>
    genNullRangeDoppler(const Eigen::Ref<const RowMatrixXcf>& echo_left,
            const Eigen::Ref<const RowMatrixXcf>& echo_right,
            const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_left,
            const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_right,
            double sr_start, double el_ang_start, double el_ang_step,
            double az_ang, std::optional<double> az_time = {}) const;

    /**
     * @return wavelength in (m)
     */
    double waveLength() const { return _wavelength; }

    /**
     * @return slant range spacing in (m)
     */
    double slantRangeSpacing() const { return _sr_spacing; }

    /**
     * @return name of antenna spherical grid type
     */
    std::string gridTypeName() const { return toStr(_ant_frame.gridType()); }

    /**
     * @return complex Hann-windowed chirp samples used as chirp ref in
     * rangecomp
     */
    Eigen::ArrayXcf chirpSampleRef() const
    {
        return Eigen::Map<const Eigen::ArrayXcf>(
                _chirp_ref.data(), _chirp_ref.size());
    }

    /**
     * @return reference epoch UTC time in ISO8601 format
     */
    std::string refEpoch() const { return _ref_epoch.isoformat(); }

    /**
     * @return reference DEM height in (m)
     */
    double demRefHeight() const { return _dem_interp.refHeight(); }

    /**
     * @return mid azimuth time of the orbit in (sec)
     */
    double midTimeOrbit() const { return _az_time_mid; }

    /**
     * @return Max EL angle spacing or resolution (rad)
     */
    double maxElSpacing() const { return _el_res_max; }

    /**
     * @return Absolute tolerance (m) in DEM height estimation used in
     * geometry transformation
     */
    double atolDEM() const { return _abs_tol_dem; }

    /**
     * @return Max iteration  in DEM height estimation used in
     * geometry transformation
     */
    int maxIterDEM() const { return _max_iter_dem; }

    /**
     * @return Absolute tolerance (-) in Newton method used in
     * estimating null location from echo null pattern
     */
    double atolNull() const { return _ftol_newton; }

    /**
     * @return Max iteration in Newton method used in
     * estimating null location from echo null pattern
     */
    int maxIterNull() const { return _max_iter_newton; }

    /**
     * @return poly-fit degree used in polyfitting echo null pattern
     */
    int polyfitDeg() const { return _polyfit_deg; }

protected:
    double _wavelength;
    double _sr_spacing;
    isce3::core::Orbit _orbit;
    isce3::core::Attitude _attitude;
    isce3::geometry::DEMInterpolator _dem_interp;
    Frame _ant_frame;
    isce3::core::Ellipsoid _ellips;
    double _el_res_max; // (rad)
    double _abs_tol_dem;
    int _max_iter_dem;
    int _polyfit_deg;

    // weighted complex chirp reference
    std::vector<std::complex<float>> _chirp_ref;
    // orbit mid azimuth time and reference epoch
    double _az_time_mid;
    isce3::core::DateTime _ref_epoch;

    // func tolerance and max iteration for Newton solver
    const double _ftol_newton {1e-5};
    const int _max_iter_newton {25};
};

}} // namespace isce3::antenna
