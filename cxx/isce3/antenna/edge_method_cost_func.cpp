#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include <isce3/antenna/edge_method_cost_func.h>
#include <isce3/except/Error.h>
#include <isce3/math/RootFind1dNewton.h>
#include <isce3/math/polyfunc.h>

namespace isce3 { namespace antenna {

std::tuple<double, double, bool, int> rollAngleOffsetFromEdge(
        const poly1d_t& polyfit_echo, const poly1d_t& polyfit_ant,
        const isce3::core::Linspace<double>& look_ang,
        std::optional<poly1d_t> polyfit_weight)
{
    // check the input arguments
    if (polyfit_echo.order != 3 || polyfit_ant.order != 3)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Requires 3rd-order poly-fit object for both "
                "Echo and Antenna!");
    constexpr double a_tol {1e-5};
    if (std::abs(polyfit_echo.mean - polyfit_ant.mean) > a_tol ||
            std::abs(polyfit_echo.norm - polyfit_ant.norm) > a_tol)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Requires same (mean, std) for Echo and Antenna Poly1d obj!");

    if (!(polyfit_echo.norm > 0.0))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Requires positive std of Echo and Antenna Poly1d obj!");

    if (polyfit_weight) {
        if (polyfit_weight->order < 0)
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                    "The order of polyfit for weights must be "
                    "at least 0 (constant weights)!");
        if (!(polyfit_weight->norm > 0.0))
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                    "Requires positive std of weight Poly1d obj!");
    }

    // create a copy polyfit objects "echo" and "ant" with zero mean and unit
    // std
    auto pf_echo_cp = polyfit_echo;
    pf_echo_cp.mean = 0.0;
    pf_echo_cp.norm = 1.0;
    auto pf_ant_cp = polyfit_ant;
    pf_ant_cp.mean = 0.0;
    pf_ant_cp.norm = 1.0;

    // declare and initialize a look angle vector
    Eigen::ArrayXd lka_vec(look_ang.size());
    for (int idx = 0; idx < look_ang.size(); ++idx)
        lka_vec(idx) = look_ang[idx];

    // create a weighting vector from look vector and weighting Poly1d
    Eigen::ArrayXd wgt_vec;
    if (polyfit_weight) {
        Eigen::Map<Eigen::ArrayXd> wgt_coef(
                polyfit_weight->coeffs.data(), polyfit_weight->coeffs.size());
        wgt_vec = isce3::math::polyval(
                wgt_coef, lka_vec, polyfit_weight->mean, polyfit_weight->norm);
        // normalize power in dB
        wgt_vec -= wgt_vec.maxCoeff();
        // convert from dB to linear power scale
        wgt_vec = Eigen::pow(10, 0.1 * wgt_vec);
    }
    // centralized and scaled the look vector based on mean/std of the echo
    // Poly1d to be used for both antenna and echo in the cost function.
    lka_vec -= polyfit_echo.mean;
    const auto std_inv = 1.0 / polyfit_echo.norm;
    lka_vec *= std_inv;

    // form some derivatives used in the cost function
    auto pf_echo_der = pf_echo_cp.derivative();
    auto pf_ant_der = pf_ant_cp.derivative();
    auto pf_ant_der2 = pf_ant_der.derivative();
    // create a memmap of the coeff for the first and second derivatives
    Eigen::Map<Eigen::ArrayXd> coef_ant_der(
            pf_ant_der.coeffs.data(), pf_ant_der.coeffs.size());
    Eigen::Map<Eigen::ArrayXd> coef_ant_der2(
            pf_ant_der2.coeffs.data(), pf_ant_der2.coeffs.size());
    Eigen::Map<Eigen::ArrayXd> coef_echo_der(
            pf_echo_der.coeffs.data(), pf_echo_der.coeffs.size());
    // form some arrays over scaled look angles for diff of first derivatives
    // and for second derivative
    auto ant_echo_der_dif_vec =
            isce3::math::polyval(coef_ant_der - coef_echo_der, lka_vec);
    auto ant_der2_vec = isce3::math::polyval(coef_ant_der2, lka_vec);

    // build cost function in the form of Poly1d object (3th order polynimal!)
    auto cf_pf = isce3::core::Poly1d(3, 0.0, 1.0);
    // fill up the coeff for the derivative of the WMSE cost function:
    // cost(ofs) = pf_wgt*(pf_echo_der(el) - pf_ant_der(el + ofs))**2
    // See section 1.1 of the cited reference.
    if (polyfit_weight) {
        auto tmp1 = wgt_vec * ant_echo_der_dif_vec;
        auto tmp2 = wgt_vec * ant_der2_vec;
        cf_pf.coeffs[0] = (tmp1 * ant_der2_vec).sum();
        cf_pf.coeffs[1] = (tmp2 * ant_der2_vec).sum() +
                          6 * pf_ant_cp.coeffs[3] * tmp1.sum();
        cf_pf.coeffs[2] = 9 * pf_ant_cp.coeffs[3] * tmp2.sum();
        cf_pf.coeffs[3] =
                18 * pf_ant_cp.coeffs[3] * pf_ant_cp.coeffs[3] * wgt_vec.sum();
    } else // no weighting
    {
        cf_pf.coeffs[0] = (ant_echo_der_dif_vec * ant_der2_vec).sum();
        cf_pf.coeffs[1] = ant_der2_vec.square().sum() +
                          6 * pf_ant_cp.coeffs[3] * ant_echo_der_dif_vec.sum();
        cf_pf.coeffs[2] = 9 * pf_ant_cp.coeffs[3] * ant_der2_vec.sum();
        cf_pf.coeffs[3] = 18 * pf_ant_cp.coeffs[3] * pf_ant_cp.coeffs[3] *
                          look_ang.size();
    }
    // form Root finding object
    auto rf_obj =
            isce3::math::RootFind1dNewton(1e-4, 20, look_ang.spacing() / 10.);
    // solve for the root/roll offset via Newton
    auto [roll, f_val, flag, n_iter] = rf_obj.root(cf_pf);
    // scale back the roll angle by std of original poly1d object
    roll *= polyfit_echo.norm;

    return {roll, f_val, flag, n_iter};
}

std::tuple<double, double, bool, int> rollAngleOffsetFromEdge(
        const poly1d_t& polyfit_echo, const poly1d_t& polyfit_ant,
        double look_ang_near, double look_ang_far, double look_ang_prec,
        std::optional<poly1d_t> polyfit_weight)
{
    if (!(look_ang_near > 0.0 && look_ang_far > 0.0 && look_ang_prec > 0.0))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "All look angles values must be positive numbers!");
    if (look_ang_near >= (look_ang_far - look_ang_prec))
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Near-range look angle shall be smaller than "
                "far one by at least one prec!");

    const auto ang_size =
            static_cast<int>((look_ang_far - look_ang_near) / look_ang_prec) +
            1;
    auto look_ang = isce3::core::Linspace<double>::from_interval(
            look_ang_near, look_ang_far, ang_size);

    return rollAngleOffsetFromEdge(
            polyfit_echo, polyfit_ant, look_ang, polyfit_weight);
}

}} // namespace isce3::antenna
