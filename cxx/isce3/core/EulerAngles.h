#pragma once
#define EIGEN_MPL2_ONLY

#include "forward.h"

#include <Eigen/Geometry>

#include "DenseMatrix.h"
#include "Quaternion.h"

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

    /** Construct from rotation matrix. */
    explicit EulerAngles(const Mat3& R)
    {
        // Isn't this abs(cos(pitch))?  Name sy seems odd.
        const double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
        if (sy >= 1.0e-6) {
            _roll = std::atan2(R(2, 1), R(2, 2));
            // No extra range vs asin(-R(2,0)) since sy is always positive,
            // but using atan2 for everything is robust to det(R) != 1.
            _pitch = std::atan2(-R(2, 0), sy);
            _yaw = std::atan2(R(1, 0), R(0, 0));
        } else {
            _roll = std::atan2(-R(1, 2), R(1, 1));
            _pitch = std::atan2(-R(2, 0), sy);
            _yaw = 0.0;
        }
    }

    /** Construct from quaternion. */
    EulerAngles(const Quaternion& q);

    /** Convert to rotation matrix. */
    Mat3 toRotationMatrix() const
    {
        using namespace Eigen;
        return (AngleAxisd(yaw(), Vector3d::UnitZ()) *
                AngleAxisd(pitch(), Vector3d::UnitY()) *
                AngleAxisd(roll(), Vector3d::UnitX()))
                .toRotationMatrix();
    }

    /** Get yaw in radians. */
    double yaw() const { return _yaw; }
    /** Get pitch in radians. */
    double pitch() const { return _pitch; }
    /** Get roll in radians. */
    double roll() const { return _roll; }

private:
    double _yaw, _pitch, _roll;
};

}} // namespace isce3::core
