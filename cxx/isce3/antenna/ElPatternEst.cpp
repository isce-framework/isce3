#include "ElPatternEst.h"

#include <cmath>

#include <isce3/core/Constants.h>
#include <isce3/except/Error.h>
#include <isce3/geometry/geometry.h>
#include <isce3/math/polyfunc.h>

#include "detail/WinChirpRgCompPow.h"

namespace isce3 { namespace antenna {

ElPatternEst::ElPatternEst(double sr_start, const isce3::core::Orbit& orbit,
        int polyfit_deg, const isce3::geometry::DEMInterpolator& dem_interp,
        double win_ped, const isce3::core::Ellipsoid& ellips,
        bool center_scale_pf)
    : _sr_start(sr_start), _orbit(orbit), _polyfit_deg(polyfit_deg),
      _dem_interp(dem_interp), _win_ped(win_ped), _ellips(ellips),
      _center_scale_pf(center_scale_pf)
{
    if (!(sr_start > 0.0))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Requires positive value for starting slant range!");
    if (polyfit_deg < 2)
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "The degree of poly fit shall be at least 2!");
    if (win_ped < 0.0 || win_ped > 1.0)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "The raised-cosine window pedestal shall be within [0, 1]!");
}

typename ElPatternEst::tuple5_t ElPatternEst::powerPattern2way(
        const Eigen::Ref<const RowMatrixXcf>& echo_mat, double sr_spacing,
        double chirp_rate, double chirp_dur, std::optional<double> az_time,
        int size_avg, bool inc_corr) const
{
    // get calibrated avreaged two-way power pattern
    auto [cal_pow, slant_range, look_ang, inc_ang] =
            _getCalibPowLinear(echo_mat, sr_spacing, chirp_rate, chirp_dur,
                    az_time, size_avg, inc_corr);
    // convert to dB
    cal_pow = 10 * Eigen::log10(cal_pow);
    // polyfit pow in dB as a function of look angles in rad with centering and
    // scaling!
    auto poly1d_obj = isce3::math::polyfitObj(
            look_ang, cal_pow, _polyfit_deg, _center_scale_pf);
    // return time-series power in dB scale
    return {cal_pow, slant_range, look_ang, inc_ang, poly1d_obj};
}

typename ElPatternEst::tuple5_t ElPatternEst::powerPattern1way(
        const Eigen::Ref<const RowMatrixXcf>& echo_mat, double sr_spacing,
        double chirp_rate, double chirp_dur, std::optional<double> az_time,
        int size_avg, bool inc_corr) const
{
    // get calibrated avreaged one-way power pattern
    auto [cal_pow, slant_range, look_ang, inc_ang] =
            _getCalibPowLinear(echo_mat, sr_spacing, chirp_rate, chirp_dur,
                    az_time, size_avg, inc_corr);
    // convert sqrt value to dB
    cal_pow = 5 * Eigen::log10(cal_pow);
    // polyfit pow in dB as a function of look angles in rad with centering and
    // scaling!
    auto poly1d_obj = isce3::math::polyfitObj(
            look_ang, cal_pow, _polyfit_deg, _center_scale_pf);
    // return time-series power in dB scale
    return {cal_pow, slant_range, look_ang, inc_ang, poly1d_obj};
}

typename ElPatternEst::tuple4_t ElPatternEst::_getCalibPowLinear(
        const Eigen::Ref<const RowMatrixXcf>& echo_mat, double sr_spacing,
        double chirp_rate, double chirp_dur, std::optional<double> az_time,
        int size_avg, bool inc_corr) const

{
    // get range sampling frequency
    const double sample_freq {isce3::core::speed_of_light / (2 * sr_spacing)};
    // form the reference weighted unit-energy complex chirp
    auto chirp_ref = detail::genRcosWinChirp(
            sample_freq, chirp_rate, chirp_dur, _win_ped);
    // calculate the mean echo power by averaging over multiple range compressed
    // range lines
    auto mean_echo_pow = detail::meanRgCompEchoPower(echo_mat, chirp_ref);
    // perform averaging over multiple range bins and partially perform relative
    // radiometric cal by compensating for 2-way range path loss
    auto [cal_avg_pow, sr] = detail::rangeCalibAvgEchoPower(
            mean_echo_pow, _sr_start, sr_spacing, size_avg);
    // estimate look angle and incidence angles per geometry/orbit at az_time
    auto [look_ang, inc_ang] = isce3::geometry::lookIncAngFromSlantRange(
            sr, _orbit, az_time, _dem_interp, _ellips);
    // as part of relative radiometric cal ,perform incidence angle correction
    // if requested
    if (inc_corr) {
        cal_avg_pow *= Eigen::tan(inc_ang);
        if (inc_ang(0) > 0.0)
            cal_avg_pow /= std::tan(inc_ang(0));
    }
    // peak normalize to get relative variation
    auto peak_pow {cal_avg_pow.maxCoeff()};
    if (peak_pow > 0.0)
        cal_avg_pow /= peak_pow;
    auto slant_range = Linspace_t(
            sr(0), sr_spacing * size_avg, static_cast<int>(sr.size()));
    return {cal_avg_pow, slant_range, look_ang, inc_ang};
}

}} // namespace isce3::antenna
