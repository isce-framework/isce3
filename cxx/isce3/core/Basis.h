//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#pragma once

#include "forward.h"

#include <isce3/except/Error.h>

#include "Common.h"
#include "DenseMatrix.h"
#include "Vector.h"

namespace isce3 { namespace core {

/** Simple class to store three-dimensional basis vectors*/
class Basis {

public:
    /** Default to identity basis */
    CUDA_HOSTDEV
    Basis() : _x0 {1, 0, 0}, _x1 {0, 1, 0}, _x2 {0, 0, 1} {}

    /** Constructor with basis vectors
     *
     * Input vectors which must be mutually orthogonal and have unit norm.
     * The optional `tol` parameter is used to check
     * \f$ \mathbf{B}^\top \mathbf{B} - \mathbf{I} = 0 \f$
     */
    CUDA_HOST
    Basis(const Vec3& x0, const Vec3& x1, const Vec3& x2, double tol = 1e-8)
        : _x0(x0), _x1(x1), _x2(x2)
    {
        auto b = toRotationMatrix();
        // Cast to array for elementwise abs() method.
        Eigen::Array33d resid = b.transpose().dot(b) - Mat3::Identity();
        if (not(resid.abs() < tol).all()) {
            // throw means this ctor is CUDA_HOST only.
            throw isce3::except::InvalidArgument(
                    ISCE_SRCINFO(), "Basis is not unitary/orthogonal");
        }
    }

    /** Geocentric TCN constructor
     * @param[in] p position vector
     * @param[in] v velocity vector */
    CUDA_HOSTDEV explicit Basis(const Vec3& p, const Vec3& v)
    {
        const Vec3 n = -p.normalized();
        const Vec3 c = n.cross(v).normalized();
        const Vec3 t = c.cross(n).normalized();
        _x0 = t;
        _x1 = c;
        _x2 = n;
    }

    /**Return first basis vector*/
    CUDA_HOSTDEV const Vec3& x0() const { return _x0; }

    /**Return second basis vector*/
    CUDA_HOSTDEV const Vec3& x1() const { return _x1; }

    /**Return third basis vector*/
    CUDA_HOSTDEV const Vec3& x2() const { return _x2; }

    /**Set the first basis vector*/
    CUDA_HOSTDEV void x0(const Vec3& x0) { _x0 = x0; }

    /**Set the second basis vector*/
    CUDA_HOSTDEV void x1(const Vec3& x1) { _x1 = x1; }

    /**Set the third basis vecot*/
    CUDA_HOSTDEV void x2(const Vec3& x2) { _x2 = x2; }

    /** \brief Project a given vector onto basis
     *
     * @param[in] vec 3D vector to project
     * @param[out] res 3D vector output
     *
     * \f[
     *      res_i = (x_i \cdot vec)
     *  \f] */
    CUDA_HOSTDEV inline Vec3 project(const Vec3& vec) const
    {
        return Vec3 {_x0.dot(vec), _x1.dot(vec), _x2.dot(vec)};
    };

    /** \brief Combine the basis with given weights
     *
     * @param[in] vec 3D vector to use as weights
     * @param[out] res 3D vector output
     *
     * \f[
     *      res = \sum_{i=0}^2 vec[i] \cdot x_i
     * \f] */
    inline void combine(const Vec3& vec, Vec3& res) const
    {
        for (int ii = 0; ii < 3; ii++) {
            res[ii] = vec[0] * _x0[ii] + vec[1] * _x1[ii] + vec[2] * _x2[ii];
        }
    };

    // NOTE explicit operator Mat3() doesn't work due to Eigen weirdness,
    // at least with GCC {8,9,10} and -std=c++17.
    /** Convert to matrix, vectors (x0, x1, x2) as columns (0, 1, 2). */
    CUDA_HOSTDEV Mat3 toRotationMatrix() const
    {
        Mat3 out;
        out.col(0) = _x0;
        out.col(1) = _x1;
        out.col(2) = _x2;
        return out;
    }

private:
    Vec3 _x0;
    Vec3 _x1;
    Vec3 _x2;
};

/** Compute velocity in inertial frame (ECI) given position and velocity in an
 *  Earth-fixed (ECF) frame.  Assumes constant rotation rate about Z-axis.
 *
 * @param[in] position      Position relative to Earth center (m).
 * @param[in] velocityECF   Velocity in Earth-fixed frame (m/s).
 */
Vec3 velocityECI(const Vec3& position, const Vec3& velocityECF);

/** Get TCN-like basis where N is perpendicular to ellipsoid.
 *
 * @param[in] x         Position (m).
 * @param[in] v         Velocity (m/s).
 * @param[in] ellipsoid Shape of planet.
 */
Basis geodeticTCN(const Vec3& x, const Vec3& v, const Ellipsoid& ellipsoid);

/** Get Euler angles between body and geodetic TCN, factoring out Earth motion.
 *
 * @param[in] q             quaternion with total rotation between body and
 *                          world frame (ECF)
 * @param[in] x             position (m)
 * @param[in] v             velocity, Earth-fixed frame (m/s)
 * @param[in] ellipsoid     Nadir defined with respect to this ellipsoid.
 */
EulerAngles factoredYawPitchRoll(const Quaternion& q, const Vec3& x,
                                 const Vec3& v, const Ellipsoid& ellipsoid);

/** Get Euler angles between body and geocentric TCN, factoring out Earth
 * motion.
 *
 * @param[in] q             quaternion with total rotation between body and
 *                          world frame (ECF)
 * @param[in] x             position (m)
 * @param[in] v             velocity, Earth-fixed frame (m/s)
 */
EulerAngles factoredYawPitchRoll(const Quaternion& q, const Vec3& x,
                                 const Vec3& v);

}} // namespace isce3::core
