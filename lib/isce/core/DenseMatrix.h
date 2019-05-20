#pragma once
#ifndef ISCE_CORE_DENSEMATRIX_H
#define ISCE_CORE_DENSEMATRIX_H

#include "Vector.h"

namespace isce { namespace core {
    template<int N> struct DenseMatrix;

    typedef DenseMatrix<3> Mat3;
}}

template<int N>
struct isce::core::DenseMatrix {

private:

    Vector<N> data[N];

public:

    CUDA_HOSTDEV constexpr DenseMatrix() {}

    CUDA_HOSTDEV constexpr DenseMatrix(std::initializer_list<Vec3> lst) {
        int i = 0;
        for (const auto& v : lst) {
            data[i++] = v;
        }
    }

    CUDA_HOSTDEV constexpr DenseMatrix(std::initializer_list<std::initializer_list<double>> lst) {
        int i = 0, j = 0;
        for (const auto& l : lst) {
            for (const auto& v : l) {
                data[i][j++] = v;
            }
            i++, j = 0;
        }
    }

    /** Matrix transposition */
    CUDA_HOSTDEV constexpr DenseMatrix<N> transpose() const {
        DenseMatrix<N> out;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                out[i][j] = (*this)[j][i];
        return out;
    }

    /** Naive O(n^3) matrix-matrix multiplication */
    CUDA_HOSTDEV explicit constexpr DenseMatrix(const DenseMatrix<N>& a, const DenseMatrix<N>& b) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double elem = 0;
                for (int k = 0; k < N; k++)
                    elem += a[i][k] * b[k][j];
                (*this)[i][j] = elem;
            }
        }
    }
    CUDA_HOSTDEV constexpr DenseMatrix<N> dot(const DenseMatrix<N>& a) const {
        return DenseMatrix(*this, a);
    }

    /** Matrix-vector multiplication
     * Calculate a matrix product and return the resulting vector */
    CUDA_HOSTDEV constexpr Vector<N> dot(const Vec3& v) const {
        Vec3 out;
        for (int i = 0; i < N; i++) {
            double elem = 0;
            for (int j = 0; j < N; j++)
                elem += (*this)[i][j] * v[j];
            out[i] = elem;
        }
        return out;
    }

    CUDA_HOSTDEV constexpr const Vector<N>& operator[](int i) const { return data[i]; }
    CUDA_HOSTDEV constexpr       Vector<N>& operator[](int i)       { return data[i]; }

    /** Compute ENU basis transformation matrix
     *  @param[in] lat Latitude in radians
     *  @param[in] lon Longitude in radians
     *  @param[out] enumat Matrix with rotation matrix */
    CUDA_HOSTDEV static Mat3 xyzToEnu(double lat, double lon) {
        using std::cos;
        using std::sin;
        return Mat3 {{{         -sin(lon),           cos(lon),       0.},
                      {-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)},
                      { cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)}}};
    }
};


#endif
