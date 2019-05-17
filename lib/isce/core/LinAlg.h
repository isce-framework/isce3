//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017
//

#pragma once
#ifndef ISCE_CORE_LINALG_H
#define ISCE_CORE_LINALG_H

#include "Cartesian.h"

namespace isce { namespace core {
    struct LinAlg;
}}

/** Simple linear algebra operations for triplets of double precision numbers*/
struct isce::core::LinAlg {
    LinAlg() = default;

    /** Multiply all elements by a scalar value */
    CUDA_HOSTDEV static void scale(Vec3& v, double s) { v = v * s; }

    /** Cross product*/
    CUDA_HOSTDEV static void cross(const Vec3& a, const Vec3& b, Vec3& c) { c = a.cross(b); }

    /** Dot product*/
    CUDA_HOSTDEV static double dot(const Vec3& a, const Vec3& b) { return a.dot(b); }

    /** Linear combination of vectors */
    CUDA_HOSTDEV static void linComb(double a, const Vec3& b, double c, const Vec3& d, Vec3& e) { e = a*b + c*d; }

    /** Matrix-matrix multiplication
     * Calculate the matrix product of two 3x3 matrices and store the
     * resulting matrix in c. */
    CUDA_HOSTDEV static void matMat(const cartmat_t& a, const cartmat_t& b, cartmat_t& c) {
        for (int i=0; i<3; i++)
            for (int j=0; j<3; ++j)
                c[i][j] = (a[i][0] * b[0][j]) + (a[i][1] * b[1][j]) + (a[i][2] * b[2][j]);
    }

    /** Matrix-vector multiplication
     * Calculate the matrix product of a 1x3 vector with a 3x3 matrix and
     * store the resulting vector in w. */
    CUDA_HOSTDEV static void matVec(const cartmat_t& t, const Vec3& v, Vec3& w) {
        for (int i=0; i<3; ++i)
            w[i] = (t[i][0] * v[0]) + (t[i][1] * v[1]) + (t[i][2] * v[2]);
    }

    /** Norm of vector */
    CUDA_HOSTDEV static double norm(const Vec3& v) { return v.norm(); }

    /** Transpose a 3x3 matrix and store the resulting matrix in b. */
    CUDA_HOSTDEV static void tranMat(const cartmat_t& a, cartmat_t& b) {
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                b[i][j] = a[j][i];
            }
        }
    }

    /** Unit vector */
    CUDA_HOSTDEV static void unitVec(const Vec3& u, Vec3& v) { v = u.unitVec(); }

    /** Compute ENU basis
     *  @param[in] lat Latitude in radians
     *  @param[in] lon Longitude in radians
     *  @param[out] enumat Matrix with rotation matrix */
    CUDA_HOSTDEV static void enuBasis(double lat, double lon, cartmat_t& enumat) {
        enumat = {{{-std::sin(lon), -std::sin(lat)*std::cos(lon), std::cos(lat)*std::cos(lon)},
                   {std::cos(lon),  -std::sin(lat)*std::sin(lon), std::cos(lat)*std::sin(lon)},
                   {0.0, std::cos(lat), std::sin(lat)}}};
    }
};

#endif
