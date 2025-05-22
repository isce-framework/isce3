#pragma once
#define EIGEN_MPL2_ONLY

#include <Eigen/Geometry>

#include "DenseMatrix.h"
#include "Quaternion.h"
#include "Vector.h"

namespace isce3 { namespace core {

/** Representation of 3-2-1 Euler angle sequence of rotations. */
class EulerAngles {
public:
    /** Construct from yaw, pitch, and roll angles.
     *
     * @param[in] y     Yaw angle (axis 3) in radians.
     * @param[in] p     Pitch angle (axis 2) in radians.
     * @param[in] r     Roll angle (axis 1) in radians.
     */
    EulerAngles(double y, double p, double r) : _yaw(y), _pitch(p), _roll(r) {}

    /**
     * Construct from rotation matrix.
     * @param[in] rotmat : Eigen or Mat3 3-D
     * rotation matrix. Must be a unitary within
     * at least 1e-6 precision.
     * @exception InvalidArgument for bad rotmat.
     */
    explicit EulerAngles(const Mat3& rotmat);

    /**
     * Construct from isce3 quaternion object.
     * @param[in] quat : isce3 Quaternion object
     */
    explicit EulerAngles(const Quaternion& quat);

    /**
     * Convert to rotation matrix.
     * @return Eigen Matrix3d or isce3 Mat3
     */
    Mat3 toRotationMatrix() const
    {
        return (Eigen::AngleAxisd(yaw(), Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch(), Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll(), Eigen::Vector3d::UnitX()))
                .toRotationMatrix();
    }

    /** Get yaw in radians. */
    double yaw() const { return _yaw; }

    /** Get pitch in radians. */
    double pitch() const { return _pitch; }

    /** Get roll in radians. */
    double roll() const { return _roll; }

    /**
     * Convert to isce3 Quaternion object.
     * @return isce3 Quaternion object
     */
    Quaternion toQuaternion() const;

    /**
     * Check if *this is approximately equals to other within
     * a desired precision
     * @param[in] other : another EulerAngles object
     * @param[in] prec (optional) : double scalar precision
     * , must be a positive value.
     * @return bool
     * @exception InvalidArgument for bad prec.
     */
    bool isApprox(const EulerAngles& other, double prec = 1e-7) const;

    /**
     * Rotate a 3-D vector by Euler object in YPR order
     * @param[in] vec : Eigen Vector3d or isce3 Vec3
     * @return rotated 3-D vector
     */
    Vec3 rotate(const Eigen::Vector3d& vec) const;

    /**
     * Overloaded in-place operator += on this
     * @param[in] rhs : EulerAngles object added to this
     * @return *this
     */
    EulerAngles& operator+=(const EulerAngles& rhs)
    {
        _yaw += rhs._yaw;
        _pitch += rhs._pitch;
        _roll += rhs._roll;
        return *this;
    }

    /**
     * Overloaded in-place operator -= on this
     * @param[in] rhs : EulerAngles object subtratced from this
     * @return *this
     */
    EulerAngles& operator-=(const EulerAngles& rhs)
    {
        _yaw -= rhs._yaw;
        _pitch -= rhs._pitch;
        _roll -= rhs._roll;
        return *this;
    }

    /**
     * Overloaded in-place operator *= on this
     * @param[in] rhs : EulerAngles object whose rotation matrix
     * is multiplied to the rotation matrix of this.
     * @return *this which is concatenation of two rotations.
     */
    EulerAngles& operator*=(const EulerAngles& rhs)
    {
        *this = EulerAngles(this->toRotationMatrix() * rhs.toRotationMatrix());
        return *this;
    }

private:
    double _yaw, _pitch, _roll;
};

// Regular/non-member functions

/**
 * Overloaded binary operator+ on EulerAngles objects
 * @param[in] lhs : EulerAngles object
 * @param[in] rhs : EulerAngles object
 * @return EulerAngles object
 */
inline EulerAngles operator+(const EulerAngles& lhs, const EulerAngles& rhs)
{
    auto sum {lhs};
    sum += rhs;
    return sum;
}

/**
 * Overloaded binary operator- on EulerAngles objects
 * @param[in] lhs : EulerAngles object
 * @param[in] rhs : EulerAngles object
 * @return EulerAngles object
 */
inline EulerAngles operator-(const EulerAngles& lhs, const EulerAngles& rhs)
{
    auto sub {lhs};
    sub -= rhs;
    return sub;
}

/**
 * Overloaded binary operator * between two EulerAngles objects
 * @param[in] lhs : EulerAngles object
 * @param[in] rhs : EulerAngles object
 * @return EulerObjects formed by concatenation of two rotations lhs*rhs.
 */
inline EulerAngles operator*(const EulerAngles& lhs, const EulerAngles& rhs)
{
    auto mul {lhs};
    mul *= rhs;
    return mul;
}

}} // namespace isce3::core
