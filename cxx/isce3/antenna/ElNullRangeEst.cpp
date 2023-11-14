#include "ElNullRangeEst.h"

#include <isce3/antenna/geometryfunc.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Vector.h>
#include <isce3/except/Error.h>
#include <isce3/math/RootFind1dNewton.h>
#include <isce3/math/polyfunc.h>

#include "ElNullAnalyses.h"
#include "detail/WinChirpRgCompPow.h"

namespace isce3 { namespace antenna {

ElNullRangeEst::ElNullRangeEst(double wavelength, double sr_spacing,
        double chirp_rate, double chirp_dur, const isce3::core::Orbit& orbit,
        const isce3::core::Attitude& attitude,
        const isce3::geometry::DEMInterpolator& dem_interp,
        const Frame& ant_frame, const isce3::core::Ellipsoid& ellips,
        double el_res, double abs_tol_dem, int max_iter_dem, int polyfit_deg)
    : _wavelength(wavelength), _sr_spacing(sr_spacing), _orbit(orbit),
      _attitude(attitude), _dem_interp(dem_interp), _ant_frame(ant_frame),
      _ellips(ellips), _el_res_max(el_res), _abs_tol_dem(abs_tol_dem),
      _max_iter_dem(max_iter_dem), _polyfit_deg(polyfit_deg)
{
    // check input arguments
    if (!(sr_spacing > 0.0))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Slant-range spacing must be a positive value!");
    if (!(wavelength > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Wavelength must be a positive value!");
    if ((polyfit_deg < 2) || (polyfit_deg % 2 != 0))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Polyfit degree must be an even number larger than 2!");
    // Range sample freq in (Hz)
    auto sample_freq = isce3::core::speed_of_light / (2.0 * sr_spacing);
    // Build the Hann windowed Ref Chirp used in all rangecomp
    _chirp_ref =
            detail::genRcosWinChirp(sample_freq, chirp_rate, chirp_dur, 0.0);
    // set azimuth mid time and start date time based on orbit info
    _az_time_mid = _orbit.midTime();
    _ref_epoch = _orbit.referenceEpoch();
}

typename ElNullRangeEst::tuple_null ElNullRangeEst::genNullRangeDoppler(
        const Eigen::Ref<const RowMatrixXcf>& echo_left,
        const Eigen::Ref<const RowMatrixXcf>& echo_right,
        const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_right, double sr_start,
        double el_ang_start, double el_ang_step, double az_ang,
        std::optional<double> az_time) const
{
    // check input arguments in terms of size and value.
    // Note some will be checked via other functions used below
    if (!(sr_start > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Start slant range must be a positive value!");
    const auto num_rgl = echo_left.rows();
    const auto num_rgb = echo_left.cols();
    if (echo_right.cols() != num_rgb || echo_right.rows() != num_rgl)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Size mismatch between two echo matrices left and right!");

    // check azimuth time to be within orbit start/end time
    // if az time for echoes is not available then use that of mid orbit time.
    if (az_time) {
        if ((*az_time < _orbit.startTime()) || (*az_time > _orbit.endTime()))
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                    "Echo azimuth time is out of range of orbit time!");
    } else
        *az_time = _az_time_mid;
    // pick EL cut patterns within peak-to-peak power gains of two adjacent
    // beams, resample/interpolate the picked values into finer el-resolution if
    // necessary based on min (_el_res_max, el_ang_step) and store their
    // conjugate versions as final complex weighting coeffs "coef_left" and
    // "coef_right" as a function EL angle vector "el_ang_vec" in (rad).
    auto [coef_left, coef_right, el_ang_vec] = genAntennaPairCoefs(
            el_cut_left, el_cut_right, el_ang_start, el_ang_step, _el_res_max);

    // form NUll pattern in antenna EL domain and locate its min location in EL,
    // This is the expected/ideal/knowldege of null location obtained purely
    // from antenna patterns. Get its magnitude in (linear)
    auto [el_null_ant, idx_null_ant, mag_null_ant, pow_pat_null_ant] =
            locateAntennaNull(coef_left, coef_right, el_ang_vec);

    // get position, velocity in ECEF at echo azimuth time
    isce3::core::Vec3 pos_ecef, vel_ecef;
    _orbit.interpolate(&pos_ecef, &vel_ecef, *az_time);
    // get attitude quaternions from ANT to ECEF at echo azimuth time
    auto quat_ant2ecef = _attitude.interpolate(*az_time);

    // convert uniform el angles to (non-uniform) slant ranges and dopplers
    auto [sr_el_vec, dop_el_vec, conv_flag_geom_ant] =
            ant2rgdop(el_ang_vec.matrix(), az_ang, pos_ecef, vel_ecef,
                    quat_ant2ecef, _wavelength, _dem_interp, _abs_tol_dem,
                    _max_iter_dem, _ant_frame, _ellips);
    // get expected slant range and Doppler for ideal/expected/antenna null
    auto sr_null_ant = sr_el_vec(idx_null_ant);
    auto dop_null_ant = dop_el_vec(idx_null_ant);

    // form noramlized averaged echo null power (linear) as a function slant
    // ranges (m) and vector of indcies used for mapping slant range to
    // respective antenna EL angles (rad) "el_ang_vec"
    auto [echo_null_pow_vec, sr_null_echo_vec, idx_null_echo_vec] =
            formEchoNull(_chirp_ref, echo_left, echo_right, sr_start,
                    _sr_spacing, coef_left, coef_right, sr_el_vec);

    // check min value of normalized echo null power pattern to make sure it's
    // larger than ideal peak-normalized antenna null value due to several
    // practical reasons such as additive/multiplicative noise, scene
    // reflectivity var, imperfect weighting coeff, etc.
    if (!(echo_null_pow_vec.minCoeff() > mag_null_ant))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "The minval of normalized echo null is not "
                "larger than that of antenna!");

    // form array of antenna null power pattern as well as EL angles (rad) for
    // echo null power pattern matching slant range "sr_null_echo_vec"
    Eigen::ArrayXd el_null_echo_vec(idx_null_echo_vec.size());
    Eigen::ArrayXd ant_null_pow_vec(idx_null_echo_vec.size());
    for (std::size_t idx = 0; idx < idx_null_echo_vec.size(); ++idx) {
        auto idx_el_valid = idx_null_echo_vec(idx);
        el_null_echo_vec(idx) = el_ang_vec(idx_el_valid);
        ant_null_pow_vec(idx) = pow_pat_null_ant(idx_el_valid);
    }
    // Perform polyfit of null power pattern in (dB) as a function of EL angle
    // in (rad)
    auto poly_echo_null = isce3::math::polyfitObj(
            el_null_echo_vec, 10 * echo_null_pow_vec.log10(), _polyfit_deg);

    // locate the null/min-value of echo null power by applying Newton solver
    // to the derivative of the polyfitted echo null power pattern.
    // set the initial solution to the expected one already estimated from
    // antennas! set the el angle tolerance to half of el angle resolution in
    // "el_ang_vec" in (rad). that is <= 0.5 * _el_max_res
    double el_ang_tol = 0.5 * std::fabs(el_ang_vec(1) - el_ang_vec(0));
    auto root_find_echo_null = isce3::math::RootFind1dNewton(
            _ftol_newton, _max_iter_newton, el_ang_tol);
    auto [el_null_echo, val_der_null, conv_flag_null, num_iter_null] =
            root_find_echo_null.root(poly_echo_null.derivative(), el_null_ant);
    // get null magnitude in (linear) at estimated null EL location
    auto mag_null_echo =
            std::pow(10.0, poly_echo_null.eval(el_null_echo) / 10.0);
    // in case of invalid polyfit coeffs, get approximate null power
    // directly from the echo samples!
    if (std::isnan(mag_null_echo))
        mag_null_echo = echo_null_pow_vec.abs().minCoeff();

    // get the true slant range (and doppler) at the EL location of echo null
    auto [sr_null_echo, dop_null_echo, conv_flag_geom_echo] =
            ant2rgdop(el_null_echo, az_ang, pos_ecef, vel_ecef, quat_ant2ecef,
                    _wavelength, _dem_interp, _abs_tol_dem, _max_iter_dem,
                    _ant_frame, _ellips);
    // get utc time in iso-8601 format for nulls
    auto date_time_az = _ref_epoch + *az_time;

    return {date_time_az,
            {sr_null_echo, el_null_echo, dop_null_echo, mag_null_echo},
            {sr_null_ant, el_null_ant, dop_null_ant, mag_null_ant},
            {conv_flag_null, conv_flag_geom_echo, conv_flag_geom_ant},
            {ant_null_pow_vec, echo_null_pow_vec, el_null_echo_vec}};
}

}} // namespace isce3::antenna
