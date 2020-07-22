#pragma once

#include <Eigen/Dense>

namespace isce3 {
namespace core {

/*
 * Convenience aliases for row-major Eigen datatypes
 */

template<typename T, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
using EMatrix2D = typename Eigen::Matrix<T, rows, cols, Eigen::RowMajor>;

template<typename T, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
using EArray2D = typename Eigen::Array<T, rows, cols, Eigen::RowMajor>;

} // namespace core
} // namespace isce3
