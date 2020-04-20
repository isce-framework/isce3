#pragma once

#include <Eigen/Dense>

namespace isce {
namespace core {

/*
 * Let `EMatrix` be an alias for a dynamic-sized
 * heap-allocated row-major Eigen matrix.
 */
template<typename T>
using EMatrix = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>;

template<typename T>
using EArray = typename Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>;

} // namespace core
} // namespace isce
