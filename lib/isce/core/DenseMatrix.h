#pragma once

#include "forward.h"

#include <cmath>
#define EIGEN_MPL2_ONLY
#include <Eigen/Dense>
#include "Common.h"

namespace isce { namespace core {

template<int N, typename T>
class DenseMatrix : public Eigen::Matrix<T, N, N> {
    using super_t = Eigen::Matrix<T, N, N>;
    using super_t::Matrix;
public:
    CUDA_HOSTDEV auto operator[](int i)       { return this->row(i); }
    CUDA_HOSTDEV auto operator[](int i) const { return this->row(i); }

    CUDA_HOSTDEV auto dot(const super_t& other) const {
        return *this * other;
    }

    CUDA_HOSTDEV auto dot(const Eigen::Matrix<T, N, 1>& other) const {
        return *this * other;
    }

    CUDA_HOSTDEV DenseMatrix() = default;
    CUDA_HOSTDEV constexpr DenseMatrix(
            std::initializer_list<std::initializer_list<double>> lst) {
        int i = 0, j = 0;
        for (const auto& l : lst) {
            for (const auto& v : l) {
                (*this)(i, j++) = v;
            }
            i++, j = 0;
        }
    }

    /** Matrix transposition */
    CUDA_HOSTDEV constexpr DenseMatrix<N, T> transpose() const {
        DenseMatrix<N, T> out;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                out[i][j] = (*this)[j][i];
        return out;
    }

    /** Compute ENU basis transformation matrix
     *  @param[in] lat Latitude in radians
     *  @param[in] lon Longitude in radians
     *  @param[out] enumat Matrix with rotation matrix */
    CUDA_HOSTDEV static Mat3 xyzToEnu(double lat, double lon);
};

template<int N, typename T>
CUDA_HOSTDEV Mat3 DenseMatrix<N, T>::xyzToEnu(double lat, double lon) {
    using std::cos;
    using std::sin;
    return Mat3 {{{         -sin(lon),           cos(lon),       0.},
                  {-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)},
                  { cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)}}};
}

}}
