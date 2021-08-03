#include "polyfunc.h"

namespace isce3 { namespace math {

std::tuple<Eigen::ArrayXd, double, double> polyfit(
        const Eigen::Ref<const Eigen::ArrayXd>& x,
        const Eigen::Ref<const Eigen::ArrayXd>& y, int deg, bool center_scale)
{
    // check the input sizes and values
    if (x.size() != y.size())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "Size mismatch between input vectors 'x' and 'y'");
    const int n_row = x.size();
    if (n_row < 2)
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "Size of input vectors must be at least 2!");
    if (deg < 1 || deg > (n_row - 1))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Bad value for order of the polynomial");
    // center and scale x if requested
    double mu {0};
    double sigma {1.0};
    Eigen::ArrayXd x_arr = x;
    if (center_scale) {
        mu = x_arr.mean();
        x_arr -= mu;
        sigma = std::sqrt(x_arr.abs2().mean());
        if (sigma > 0.0)
            x_arr /= sigma;
    }
    // form A matrix in Ax=b
    const auto n_col = deg + 1;
    Eigen::MatrixXd mat(n_row, n_col);
    for (int idx = 0; idx < n_col; ++idx)
        mat.col(idx) = x_arr.pow(idx);
    // solve Ax=b by using full pivoting householder QR approach
    return {mat.fullPivHouseholderQr().solve(y.matrix()), mu, sigma};
}

isce3::core::Poly1d polyfitObj(const Eigen::Ref<const Eigen::ArrayXd>& x,
        const Eigen::Ref<const Eigen::ArrayXd>& y, int deg, bool center_scale)
{
    auto [coef, x_mean, x_std] = polyfit(x, y, deg, center_scale);
    auto poly_obj = isce3::core::Poly1d(coef.size() - 1, x_mean, x_std);
    poly_obj.coeffs =
            std::vector<double>(coef.data(), coef.data() + coef.size());
    return poly_obj;
}

/**
 * @internal
 * Honer method of polynomial evaluation used in polyval()
 * <a href="https://en.wikipedia.org/wiki/Horner's_method" target="_blank">
 * See Horner's method</a>
 * @param[in] coef a vector of polynomial coeff in ascending order.
 * @param[in] x desired x value to be evaluated.
 * @param[in] mean is mean value or offset for centralizing input.
 * @param[in] std is std value or divider for scaling input.
 * @return evaluated scalar value at "x".
 */
static double _horner_polyval(const Eigen::Ref<const Eigen::ArrayXd>& coef,
        double x, double mean, double std)
{
    // centralized and scaled "x"
    auto x_c = (x - mean) / std;
    // use Horner's method
    auto i = coef.size() - 1;
    double y {coef(i)};
    while (i > 0) {
        y = x_c * y + coef(--i);
    }
    return y;
}

double polyval(const Eigen::Ref<const Eigen::ArrayXd>& coef, double x,
        double mean, double std)
{
    // check the input
    if (!(std > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "STD must be positive value!");
    if (coef.size() < 1)
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "Size of polynomial coeff vector must be at least 1!");
    // call Hoerner method
    return _horner_polyval(coef, x, mean, std);
}

Eigen::ArrayXd polyval(const Eigen::Ref<const Eigen::ArrayXd>& coef,
        const Eigen::Ref<const Eigen::ArrayXd>& x, double mean, double std)
{
    // check the input
    if (!(std > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "STD must be positive value!");
    if (coef.size() < 1)
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "Size of polynomial coeff vector must be at least 1!");
    if (x.size() < 1)
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "Size of array 'x' must be at least 1!");
    // evaluated array
    Eigen::ArrayXd y(x.size());
    for (Eigen::Index idx = 0; idx < x.size(); ++idx)
        y(idx) = _horner_polyval(coef, x(idx), mean, std);

    return y;
}

Eigen::ArrayXd polyder(const Eigen::Ref<const Eigen::ArrayXd>& coef, double std)
{
    // check the input
    if (coef.size() < 2)
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "Size of polynomial coeff vector must be at least 2!");
    if (!(std > 0.0))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "STD must be positive value!");
    // get the scaling factors based on powers for forming derivative
    const int size_out = coef.size() - 1;
    Eigen::ArrayXd der = Eigen::ArrayXd::LinSpaced(size_out, 1, size_out);
    // scale by std
    der /= std;
    // get a new truncated scaled coeffs representing first derivative of
    // orignal coeff
    return der * coef.tail(size_out);
}

}} // namespace isce3::math
