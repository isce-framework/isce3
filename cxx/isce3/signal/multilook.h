#pragma once

#include <isce3/core/EMatrix.h>

namespace isce3 {
namespace signal {

/**
 * @brief Multilooks an input Eigen::Array by taking the
 * weighted average of contributions to each pixel.
 *
 * Note that input pixels with total weight zero
 * will be NaN in the output array.
 *
 * @param[in] input     The input array to multilook
 * @param[in] row_looks The number of looks in the vertical direction
 * @param[in] col_looks The number of looks in the horizontal direction
 * @param[in] weights   The array of weights to apply to the input
 * @returns The multilooked output, as an Eigen::Array
 */
template<typename EigenT1, typename EigenT2>
auto multilookWeightedAvg(const EigenT1& input, int row_looks, int col_looks,
                          const EigenT2& weights)
{

    const auto nrows = input.rows() / row_looks;
    const auto ncols = input.cols() / col_looks;

    using value_type = typename EigenT1::value_type;
    isce3::core::EArray2D<value_type> output(nrows, ncols);

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {

            const auto cells = input.block(row * row_looks, col * col_looks,
                                           row_looks, col_looks);
            const auto wgts = weights.block(row * row_looks, col * col_looks,
                                            row_looks, col_looks);

            output(row, col) = (cells * wgts).sum() / wgts.sum();
        }
    }

    return output;
}

/**
 * @brief Multilooks an input Eigen::Array
 * by summing contributions to each pixel.
 *
 * @param[in] input     The input array to multilook
 * @param[in] row_looks The number of looks in the vertical direction
 * @param[in] col_looks The number of looks in the horizontal direction
 * @returns The multilooked output, as an Eigen::Array
 */
template<typename EigenType>
auto multilookSummed(const EigenType& input, int row_looks, int col_looks)
{

    const auto nrows = input.rows() / row_looks;
    const auto ncols = input.cols() / col_looks;

    using value_type = typename EigenType::value_type;
    isce3::core::EArray2D<value_type> output(nrows, ncols);

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {

            output(row, col) = input.block(row * row_looks, col * col_looks,
                                           row_looks, col_looks)
                                       .sum();
        }
    }

    return output;
}

/**
 * @brief Multilooks an input Eigen::Array
 * by averaging contributions to each pixel.
 *
 * @param[in] input     The input array to multilook
 * @param[in] row_looks The number of looks in the vertical direction
 * @param[in] col_looks The number of looks in the horizontal direction
 * @returns The multilooked output, as an Eigen::Array
 */
template<typename EigenType>
auto multilookAveraged(const EigenType& input, int row_looks, int col_looks)
{

    const auto nrows = input.rows() / row_looks;
    const auto ncols = input.cols() / col_looks;

    using value_type = typename EigenType::value_type;
    isce3::core::EArray2D<value_type> output(nrows, ncols);

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {

            output(row, col) = input.block(row * row_looks, col * col_looks,
                                           row_looks, col_looks)
                                       .mean();
        }
    }

    return output;
}

/**
 * @brief Multilooks an input Eigen::Array by taking the average of
 * contributions to each pixel, while masking out a provided constant
 * no-data value.
 *
 * Note that input areas that are completely masked will be NaN in
 * the output array.
 *
 * @param[in] input     The input array to multilook
 * @param[in] row_looks The number of looks in the vertical direction
 * @param[in] col_looks The number of looks in the horizontal direction
 * @param[in] nodata    The value which will be masked out of the input array
 * @returns The multilooked output, as an Eigen::Array
 */
template<typename EigenInput>
auto multilookNoData(const EigenInput& input, int row_looks, int col_looks,
                     const typename EigenInput::value_type nodata)
{

    auto upcast_bool = [](const bool b) {
        typename EigenInput::value_type ret = b ? 1 : 0;
        return ret;
    };

    auto weights = (input != nodata).unaryExpr(upcast_bool);

    return multilookWeightedAvg(input, row_looks, col_looks, weights);
}

/**
 * @brief Multilooks an input Eigen::Array by taking the sum of
 * contributions to each pixel, each raised to a given constant exponent.
 *
 * @param[in] input     The input array to multilook
 * @param[in] row_looks The number of looks in the vertical direction
 * @param[in] col_looks The number of looks in the horizontal direction
 * @param[in] exponent  The power to which each element will be raised
 * @returns The multilooked output, as an Eigen::Array
 */
template<typename EigenInput>
auto multilookPow(const EigenInput& input, int row_looks, int col_looks,
                  const int exponent)
{

    return multilookAveraged(input.abs().pow(exponent), row_looks, col_looks);
}

} // namespace signal
} // namespace isce3
