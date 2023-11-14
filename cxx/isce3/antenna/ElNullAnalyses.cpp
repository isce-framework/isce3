#include "ElNullAnalyses.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <isce3/core/Interp1d.h>
#include <isce3/core/Kernels.h>
#include <isce3/except/Error.h>
#include <isce3/focus/RangeComp.h>

namespace isce3 { namespace antenna {

Eigen::ArrayXcd linearInterpComplex1d(
        const Eigen::Ref<const Eigen::ArrayXd>& x0, const Linspace_t& x,
        const Eigen::Ref<const Eigen::ArrayXcd>& y)
{
    // check the input arguments
    if (!(x.spacing() > 0.0))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "The argument 'x' must be monotonically increasing!");
    if (x.size() != y.size())
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "The arguments 'x' and 'y' must have the same size!");
    if ((x0.minCoeff() < x.first()) || (x0.maxCoeff() > x.last()))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "The 'x0' values are out of range of 'x'!");

    // form the linear kernel
    auto linear_kernel = isce3::core::LinearKernel<double>();
    // allocate interpolated output vector
    Eigen::ArrayXcd y0(x0.size());
#pragma omp parallel for
    for (Eigen::Index idx = 0; idx < x0.size(); ++idx) {
        auto x_int = (x0(idx) - x.first()) / x.spacing();
        y0(idx) = isce3::core::interp1d(
                linear_kernel, y.data(), y.size(), 1, x_int, false);
    }
    return y0;
}

tuple_ant genAntennaPairCoefs(
        const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_right,
        double el_ang_start, double el_ang_step,
        std::optional<double> el_res_max)
{
    // check input arguments
    if (!(el_ang_step > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "EL angle step must be a positive value!");

    if (el_res_max) {
        if (!(*el_res_max > 0.0))
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                    "Max EL angle resolution must be a positive value!");
    } else
        *el_res_max = el_ang_step;

    const auto cut_size {el_cut_left.size()};
    if (el_cut_right.size() != cut_size)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Size mismatch between two EL-cut vectors left and right!");

    // pick EL cut patterns within peak-to-peak power gains of two adjacent
    // beams, resample/interpolate the picked values into finer el-resolution if
    // necessary based on min (_el_res_max, el_ang_step) and store their
    // conjugate versions as final weighting coefs as a function EL angle vector
    // "el_ang_vec" in (rad).
    Eigen::Index idx_peak_left;
    Eigen::Index idx_peak_right;
    el_cut_left.abs().maxCoeff(&idx_peak_left);
    el_cut_right.abs().maxCoeff(&idx_peak_right);
    if (idx_peak_left >= idx_peak_right)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Wrong order or incorrect EL cuts of left and right beams!");

    // take out one sample on each side (that is to exclude the peaks)
    idx_peak_left += 1;
    idx_peak_right -= 1;
    // check to have at least 3 points!
    auto num_idx_p2p = idx_peak_right - idx_peak_left + 1;
    if (num_idx_p2p < 3)
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "There is not enough samples/separation "
                "between peaks of EL cuts!");

    // form output uniform EL angle vector within [idx_peak_left,
    // idx_peak_right]
    auto el_space = std::min(*el_res_max, el_ang_step);
    auto el_start = idx_peak_left * el_ang_step + el_ang_start;
    auto el_stop = idx_peak_right * el_ang_step + el_ang_start;
    auto num_idx = static_cast<Eigen::Index>(
            std::round((el_stop - el_start) / el_space) + 1);

    Eigen::ArrayXd el_vec =
            Eigen::ArrayXd::LinSpaced(num_idx, el_start, el_stop);

    // interpolate the complex coeff within [idx_peak_left, idx_peak_right] if
    // necessary
    if (*el_res_max < el_ang_step)
    // perform linear interpolation of complex coeffs
    {
        auto el_ang_linspace = Linspace_t(el_ang_start, el_ang_step, cut_size);
        Eigen::ArrayXcd coef_left = Eigen::conj(
                linearInterpComplex1d(el_vec, el_ang_linspace, el_cut_left));
        Eigen::ArrayXcd coef_right = Eigen::conj(
                linearInterpComplex1d(el_vec, el_ang_linspace, el_cut_right));
        return std::make_tuple(coef_left, coef_right, el_vec);
    }
    // no interpolation
    return std::make_tuple(
            Eigen::conj(el_cut_left.segment(idx_peak_left, num_idx)),
            Eigen::conj(el_cut_right.segment(idx_peak_left, num_idx)), el_vec);
}

std::tuple<double, Eigen::Index, double, Eigen::ArrayXd> locateAntennaNull(
        const Eigen::Ref<const Eigen::ArrayXcd>& coef_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& coef_right,
        const Eigen::Ref<const Eigen::ArrayXd>& el_ang_vec)
{
    // antenna null power pattern = |left**2 - right**2|/(left**2 + right**2)
    // Its min is the null location.
    auto pow_ant_left = coef_left.abs2();
    auto pow_ant_right = coef_right.abs2();
    Eigen::ArrayXd pow_null_ant = (pow_ant_left - pow_ant_right).abs() /
                                  (pow_ant_left + pow_ant_right);
    // locate the null from power pattern
    Eigen::Index idx_null_ant;
    double min_val = pow_null_ant.minCoeff(&idx_null_ant);
    double max_val = pow_null_ant.maxCoeff();
    if (!(max_val > min_val))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "The antenna null pattern does not have a dip!");
    // get the peak-normalized null value in linear scale
    double val_null {min_val / max_val};
    // get EL angle of the null in (rad)
    double el_null_ant = el_ang_vec(idx_null_ant);
    // peak normalized antenna null power pattern
    pow_null_ant /= max_val;
    return {el_null_ant, idx_null_ant, val_null, std::move(pow_null_ant)};
}

tuple_echo formEchoNull(const std::vector<std::complex<float>>& chirp_ref,
        const Eigen::Ref<const RowMatrixXcf>& echo_left,
        const Eigen::Ref<const RowMatrixXcf>& echo_right, double sr_start,
        double sr_spacing, const Eigen::Ref<const Eigen::ArrayXcd>& coef_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& coef_right,
        const Eigen::Ref<const Eigen::ArrayXd>& sr_coef)
{
    // form rangecomp obj used for all range comp in forming echo null pattern
    auto rgc_obj = isce3::focus::RangeComp(chirp_ref, echo_left.cols(), 1,
            isce3::focus::RangeComp::Mode::Valid);
    // final number of range bins for the echo after range comp
    const auto num_rgb_echo = rgc_obj.outputSize();
    // form uniform slant range (m) vector for only valid part of final
    // rangecomp echo
    const auto sr_stop = sr_start + (num_rgb_echo - 1) * sr_spacing;
    Eigen::ArrayXd sr_echo_valid =
            Eigen::ArrayXd::LinSpaced(num_rgb_echo, sr_start, sr_stop);

    // get limited [first, last] indices by intersecting uniform slant ranges
    // from only valid-mode rangecomp echo with that of non-uniform one from
    // antenna weighting coefs
    auto [idx_coef_first, idx_coef_last, idx_echo_first, idx_echo_last] =
            detail::intersect_nearest(sr_coef, sr_echo_valid);
    const auto size_coef = idx_coef_last - idx_coef_first + 1;
    const auto num_rgb_null = idx_echo_last - idx_echo_first + 1;
    // now get array of relative indices for mapping uniform sr_echo[first,
    // last] to non-uniform  sr_coeff[first, last]
    auto idx_coef_vec =
            detail::locate_nearest(sr_coef.segment(idx_coef_first, size_coef),
                    sr_echo_valid.segment(idx_echo_first, num_rgb_null));
    // add the first index to all values for absolute mapping from echo[first,
    // last] to antenna coeff.
    idx_coef_vec += idx_coef_first;

    // get limited coefs left/right within non-uniform indices "idx_coef_vec"
    // related to uniform range bins of echo.
    Eigen::ArrayXcd coef_left_limit(num_rgb_null);
    Eigen::ArrayXcd coef_right_limit(num_rgb_null);
    for (Eigen::Index idx = 0; idx < idx_coef_vec.size(); ++idx) {
        coef_left_limit(idx) = coef_left(idx_coef_vec(idx));
        coef_right_limit(idx) = coef_right(idx_coef_vec(idx));
    }

    // initialize the peak-normalized averaged echo null power
    Eigen::ArrayXd pow_null_avg = Eigen::ArrayXd::Zero(num_rgb_null);
    // allocate rangeline vector for range compression of left/right echoes
    Eigen::ArrayXcf rgc_left(num_rgb_echo);
    Eigen::ArrayXcf rgc_right(num_rgb_echo);
    // allocate double precision lines for left and right weighted rangecomp
    // echo within null formation part only
    Eigen::ArrayXcd line_left(num_rgb_null);
    Eigen::ArrayXcd line_right(num_rgb_null);
    // loop over range lines /pulses
    for (Eigen::Index pulse = 0; pulse < echo_left.rows(); ++pulse) {
        // range compression of echoes left/right
        rgc_obj.rangecompress(rgc_left.data(), echo_left.row(pulse).data());
        rgc_obj.rangecompress(rgc_right.data(), echo_right.row(pulse).data());

        // get the power and add up its double precision version
        line_left = rgc_left.segment(idx_echo_first, num_rgb_null)
                            .cast<std::complex<double>>() *
                    coef_left_limit;
        line_right = rgc_right.segment(idx_echo_first, num_rgb_null)
                             .cast<std::complex<double>>() *
                     coef_right_limit;

        // form the null power to be averaged over range lines
        pow_null_avg +=
                (line_left - line_right).abs() / (line_left + line_right).abs();
    }
    auto max_pow_null = pow_null_avg.maxCoeff();
    if (!(max_pow_null > pow_null_avg.minCoeff()))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "The echo null power pattern does not have a dip!");
    // peak-normalized power pattern
    pow_null_avg /= max_pow_null;

    // power (linear), slant range (m) , index for EL angles (-)
    return std::make_tuple(pow_null_avg,
            sr_coef.segment(idx_coef_vec(0), num_rgb_null), idx_coef_vec);
}

}} // namespace isce3::antenna
