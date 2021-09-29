#pragma once

#include "forward.h"

#include <cmath>
#define EIGEN_MPL2_ONLY
#include <Eigen/Dense>

#include "Common.h"
#include "Vector.h"

namespace isce3 { namespace core {

template<int N, typename T>
class DenseMatrix : public Eigen::Matrix<T, N, N> {
    using super_t = Eigen::Matrix<T, N, N>;
    using super_t::super_t;

    static_assert(N > 0);

public:
    DenseMatrix() = default;
    CUDA_HOSTDEV auto operator[](int i)       { return this->row(i); }
    CUDA_HOSTDEV auto operator[](int i) const { return this->row(i); }

    CUDA_HOSTDEV auto dot(const DenseMatrix& other) const
    {
        return *this * other;
    }

    CUDA_HOSTDEV auto dot(const Vector<N, T>& other) const
    {
        return *this * other;
    }

// Backport Eigen 3.4.0's initializer_list constructor
#if !EIGEN_VERSION_AT_LEAST(3, 4, 0)
    CUDA_HOSTDEV explicit constexpr DenseMatrix(
            std::initializer_list<std::initializer_list<T>> lst) {
        int i = 0, j = 0;
        for (const auto& l : lst) {
            for (const auto& v : l) {
                (*this)(i, j++) = v;
            }
            i++, j = 0;
        }
    }
#endif

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

    /** Compute ENU basis inverse transformation matrix
     *  @param[in] lat Latitude in radians
     *  @param[in] lon Longitude in radians
     *  @param[out] enumat Matrix with rotation matrix */
    CUDA_HOSTDEV static Mat3 enuToXyz(double lat, double lon);
};

template<int N, typename T>
CUDA_HOSTDEV Mat3 DenseMatrix<N, T>::xyzToEnu(double lat, double lon) {
    using std::cos;
    using std::sin;
    return Mat3 {{{         -sin(lon),           cos(lon),       0.},
                  {-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)},
                  { cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)}}};
}

template<int N, typename T>
CUDA_HOSTDEV Mat3 DenseMatrix<N, T>::enuToXyz(double lat, double lon)
{
    using std::cos;
    using std::sin;
    return Mat3 {{{-sin(lon), -sin(lat) * cos(lon), cos(lat) * cos(lon)},
                  {cos(lon), -sin(lat) * sin(lon), cos(lat) * sin(lon)},
                  {0, cos(lat), sin(lat)}}};
}

// XXX
// These overloads are a workaround to resolve an issue observed with certain
// Eigen & CUDA version combinations where matrix-matrix and matrix-vector
// multiplication produced incorrect results (or raised "illegal memory access"
// errors in debug mode).
template<int N, typename T>
CUDA_HOSTDEV auto operator*(
        const DenseMatrix<N, T>& a, const DenseMatrix<N, T>& b)
{
    DenseMatrix<N, T> out;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out(i, j) = a.row(i).dot(b.col(j));
        }
    }
    return out;
}

template<int N, typename T>
CUDA_HOSTDEV auto operator*(const DenseMatrix<N, T>& m, const Vector<N, T>& v)
{
    Vector<N, T> out;
    for (int i = 0; i < N; ++i) {
        out[i] = m.row(i).dot(v);
    }
    return out;
}

}}
