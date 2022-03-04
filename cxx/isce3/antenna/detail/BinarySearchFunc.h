/** @file BinarySearchFunc.h
 * Functions for binary search of sorted Eigen Arrays
 */
#pragma once

#include <tuple>
#include <vector>

#include <Eigen/Dense>

namespace isce3 { namespace antenna { namespace detail {

// aliases
using tuple4i_t =
        std::tuple<Eigen::Index, Eigen::Index, Eigen::Index, Eigen::Index>;

using ArrayXui = Eigen::Array<Eigen::Index, Eigen::Dynamic, 1>;

// functions

/**
 * Python-like bisect_left for an array sorted in ascending order
 * @param[in] x is a sorted array to be searched
 * @param[in] x0 is the value to search for
 * @return first index to the array "x" where x[idx] >= x0
 * @note The output will be limited to [0, x.size - 1] for out of range value.
 **/
Eigen::Index bisect_left(const Eigen::Ref<const Eigen::ArrayXd>& x, double x0);

/**
 * Python-like bisect_right for sorted array
 * @param[in] x is a sorted array to be searched
 * @param[in] x0 is the value to search for
 * @return index to the array "x" where x[idx] > x0
 * @note The output will be limited to [0, x.size - 1] for out of range value.
 */
Eigen::Index bisect_right(const Eigen::Ref<const Eigen::ArrayXd>& x, double x0);

/**
 * Locate the nearest neighbor to a value x0 in a sorted array x via binary
 * search
 * @param[in] x is a sorted array to be searched
 * @param[in] x0 is the value to search for in "x"
 * @return index to the array "x" where abs(x[idx] - x0) is min.
 * @note The output will be limited to [0, x.size - 1] for out of range value.
 */
Eigen::Index locate_nearest(
        const Eigen::Ref<const Eigen::ArrayXd>& x, double x0);

/**
 * Locate the nearest neighbor to an array of values, x0, in a sorted  array x
 * via binary search
 * @param[in] x is a sorted array to be searched
 * @param[in] x0 is the array of values to search for in "x"
 * @return Array of indices to the array "x" with the same size as that of x0.
 * @note The indices will all be limited to within [0, x.size - 1].
 */
ArrayXui locate_nearest(const Eigen::Ref<const Eigen::ArrayXd>& x,
        const Eigen::Ref<const Eigen::ArrayXd>& x0);

/**
 * Intersect two sorted arrays x1 and x2 arrays to find the indices [first,last]
 * for approximate overlap portion defined in terms of their nearest start/end
 * values.
 * @param[in] x1 is a sorted array
 * @param[in] x2 is a sorted array
 * @return first index to array "x1"
 * @return last index (inclusive) to array "x1"
 * @return first index to array "x2"
 * @return last index (inclusive) to array "x2"
 */
tuple4i_t intersect_nearest(const Eigen::Ref<const Eigen::ArrayXd>& x1,
        const Eigen::Ref<const Eigen::ArrayXd>& x2);

}}} // namespace isce3::antenna::detail
