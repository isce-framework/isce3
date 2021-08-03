#pragma once

#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <isce3/core/EMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly1d.h>
#include <isce3/geometry/DEMInterpolator.h>

namespace isce3 { namespace antenna {

/**
 * A class for estimating one-way or two-way elevation (EL) power
 * pattern from 2-D raw echo data over quasi-homogenous scene
 * and provide approximate look (off-nadir or EL) angle  and
 * ellipsoidal incidence angle (no local slope).
 * The final power in dB scale is fit into N-order polynomials
 * as a function of look angle in radians.
 * Required relative radiometric calibration from \f$\beta^{0}\f$ to
 * \f$\gamma^{0}\f$ is done approximately by using slant ranges and
 * ellipsoidal incidence angle.
 */
class ElPatternEst {
    // aliases
public:
    using RowMatrixXcf = isce3::core::EMatrix2D<std::complex<float>>;

protected:
    using Linspace_t = isce3::core::Linspace<double>;
    using tuple4_t = std::tuple<Eigen::ArrayXd, isce3::core::Linspace<double>,
            Eigen::ArrayXd, Eigen::ArrayXd>;
    using tuple5_t = std::tuple<Eigen::ArrayXd, isce3::core::Linspace<double>,
            Eigen::ArrayXd, Eigen::ArrayXd, isce3::core::Poly1d>;

public:
    // constructors
    /**
     * A constructor with full input arguments.
     * @param[in] sr_start start slant range in (m)
     * @param[in] orbit isce3 Orbit object
     * @param[in] polyfit_deg (optional) polyfit degree of polynomial
     * for polyfitting estimated power patterns. The value must be > 1.
     * Default is 6.
     * @pram[in] dem_interp (optional) isce3 DEMInterpolator object.
     * Default is global 0.0 (m) height (reference ellipsoid).
     * Note that simply averaged DEM will be used. No local slope is taken into
     * account!
     * @param[in] win_ped (optional) Raised-cosine window pedestal. A value
     * within [0, 1]. Default is Hann
     * window.https://en.wikipedia.org/wiki/Hann_function <a
     * href="https://en.wikipedia.org/wiki/Hann_function" target="_blank">See
     * raised cosine window</a>.
     * @param[in] ellips (optional) isce3 Ellipsoid object. Default is WGS84.
     * @param[in] center_scale_pf (optional) whether or not use center and
     * scaling in polyfit. Default is false.
     * @exception InvalidArgument
     */
    ElPatternEst(double sr_start, const isce3::core::Orbit& orbit,
            int polyfit_deg = 6,
            const isce3::geometry::DEMInterpolator& dem_interp = {},
            double win_ped = 0.0, const isce3::core::Ellipsoid& ellips = {},
            bool center_scale_pf = false);

    /**
     * A more concise constructor of key inputs.
     * @param[in] sr_start  start slant range in (m)
     * @param[in] orbit  isce3 Orbit object
     * @param[in] dem_interp  isce3 DEMInterpolator object.
     * @exception InvalidArgument
     */
    ElPatternEst(double sr_start, const isce3::core::Orbit& orbit,
            const isce3::geometry::DEMInterpolator& dem_interp)
        : ElPatternEst(sr_start, orbit, 6, dem_interp, 0.0,
                  isce3::core::Ellipsoid(), false)
    {}

    // methods
    /**
     * Estimated averaged two-way time-series power pattern in Elevation from
     * 2-D raw echo data uniformly sampled in slant range direction. Uniform
     * sampling in azimuth direction is not required! Note that it is assumed
     * the data either has no TX gap (bad values) or its TX gap (bad values)
     * have been already replaced by meaningful data.
     * @param[in] echo_mat raw echo matrix, a row-major Eigen matrix of type
     * complex float. The rows represent range lines. The matrix shape is pulses
     * (azimuth bins) by range bins.
     * @param[in] sr_spacing slant range spacing in (m).
     * @param[in] chirp_rate transmit chirp rate in (Hz/sec).
     * @param[in] chirp_dur transmit chirp duration in (sec).
     * @param[in] az_time (optional) relative azimuth time in seconds w.r.t
     * reference epoch time of orbit object. Default is the mid orbit time if
     * not specified or if set to {} or std::nullopt.
     * @param[in] size_avg (optional) the block size for averaging in slant
     * range direction. Default is 8.
     * @param[in] inc_corr (optional) whether or not apply correction for
     * incidence angles calculated at a certain mean DEM height wrt reference
     * ellipsoid w/o taking into account any local slope! Default is true.
     * @return eigen vector of times-series range-averaged peak-normalized
     * 2-way power pattern in (dB) which is uniformly sampled in slant range
     * with new range spacing defined by "sr_spacing * size_avg".
     * @return slant ranges in meters in the form of isce3 Linspace object
     * @return look angles vector in radians.
     * @return ellipsoidal incidence angles vector in radians.
     * @return isce3 Poly1d object mapping power in dB as a function
     * look angle in radians
     * @exception InvalidArgument, RuntimeError
     * @see powerPattern1way()
     */
    tuple5_t powerPattern2way(const Eigen::Ref<const RowMatrixXcf>& echo_mat,
            double sr_spacing, double chirp_rate, double chirp_dur,
            std::optional<double> az_time = {}, int size_avg = 8,
            bool inc_corr = true);

    /**
     * Estimated averaged one-way time-series power pattern in Elevation from
     * 2-D raw echo data uniformly sampled in slant range direction. Uniform
     * sampling in azimuth direction is not required! Note that  it is assumed
     * the data either has no TX gap (bad values) or its TX gap (bad values)
     * have been already replaced by meaningful data.
     * @param[in]  echo_mat raw echo matrix, a row-major Eigen matrix of type
     * complex float. The rows represent range lines. The matrix shape is pulses
     * (azimuth bins) by range bins.
     * @param[in] sr_spacing slant range spacing in (m).
     * @param[in] chirp_rate transmit chirp rate in (Hz/sec).
     * @param[in] chirp_dur transmit chirp duration in (sec).
     * @param[in] az_time (optional) relative azimuth time in seconds w.r.t
     * reference epoch time of orbit object. Default is the mid orbit time if
     * not specified or if set to {} or std::nullopt.
     * @param[in] size_avg (optional) the block size for averaging in slant
     * range direction. Default is 8.
     * @param[in] inc_corr (optional) whether or not apply correction for
     * incidence angles calculated at a certain mean DEM height wrt reference
     * ellipsoid w/o taking into account any local slope! Default is true.
     * @return eigen vector of times-series range-averaged peak-normalized
     * 1-way power pattern in (dB) which is uniformly sampled in slant range
     * with new range spacing defined by "sr_spacing * size_avg".
     * @return slant ranges in meters in the form of isce3 Linspace object
     * @return look angles vector in radians.
     * @return ellipsoidal incidence angles vector in radians.
     * @return isce3 Poly1d object mapping power in dB as a function
     * look angle in radians
     * @exception InvalidArgument, RuntimeError
     * @see powerPattern2way()
     */
    tuple5_t powerPattern1way(const Eigen::Ref<const RowMatrixXcf>& echo_mat,
            double sr_spacing, double chirp_rate, double chirp_dur,
            std::optional<double> az_time = {}, int size_avg = 8,
            bool inc_corr = true);

    /**
     * Get raised-cosine window pedestal set at the constructor.
     * @return window pedestal used for weighting ref chirp in
     * range compression.
     */
    double winPed() const { return _win_ped; }

    /**
     * Get degree of polyfit set at the constructor.
     * @return degree of polyfit used for 1-way or 2-way power pattern (dB)
     * as a function of look angles (rad).
     */
    int polyfitDeg() const { return _polyfit_deg; }

    /**
     * Check whether the polyfit process is centeralized and scaled.
     * @return bool
     */
    bool isCenterScalePolyfit() const { return _center_scale_pf; }

private:
    /**
     * Helper method for public methods "powPat1w" and "powPat2w"
     * @return peak-normalized calibrated averaged 2-way power pattern vector
     * in linear scale
     * @return slant ranges in meters in the form of isce3 Linspace object
     * @return look angles vector in radians.
     * @return ellipsoidal incidence angles vector in radians.
     * look angle in radians
     * @see powerPattern2way(), powerPattern1way()
     */
    tuple4_t _getCalibPowLinear(const Eigen::Ref<const RowMatrixXcf>& echo_mat,
            double sr_spacing, double chirp_rate, double chirp_dur,
            std::optional<double> az_time, int size_avg, bool inc_corr);

    // members
protected:
    // input common parameters
    double _sr_start;
    isce3::core::Orbit _orbit;
    double _polyfit_deg;
    isce3::geometry::DEMInterpolator _dem_interp;
    double _win_ped;
    isce3::core::Ellipsoid _ellips;
    bool _center_scale_pf;
};

}} // namespace isce3::antenna
