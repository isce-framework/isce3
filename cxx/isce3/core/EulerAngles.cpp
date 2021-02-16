#include "EulerAngles.h"

#include <isce3/except/Error.h>

namespace isce3 { namespace core {

EulerAngles::EulerAngles(const Quaternion& quat)
    : EulerAngles(quat.toEulerAngles())
{}

EulerAngles::EulerAngles(const Mat3& rotmat)
{
    if (!rotmat.isUnitary(1e-6))
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Requires unitary/rotation Matrix!");

    // Isn't this abs(cos(pitch))?  Name sy seems odd.
    const double sy = std::sqrt(
            rotmat(0, 0) * rotmat(0, 0) + rotmat(1, 0) * rotmat(1, 0));
    if (sy >= 1.0e-6) {
        _roll = std::atan2(rotmat(2, 1), rotmat(2, 2));
        // No extra range vs asin(-R(2,0)) since sy is always positive,
        // but using atan2 for everything is robust to det(R) != 1.
        _pitch = std::atan2(-rotmat(2, 0), sy);
        _yaw = std::atan2(rotmat(1, 0), rotmat(0, 0));
    } else {
        _roll = std::atan2(-rotmat(1, 2), rotmat(1, 1));
        _pitch = std::atan2(-rotmat(2, 0), sy);
        _yaw = 0.0;
    }
}

bool EulerAngles::isApprox(const EulerAngles& other, double prec) const
{
    if (prec <= 0.0)
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "prec must be a positive value!");
    auto isSame = [&prec](double x, double y) {
        return std::abs(x - y) <= prec;
    };
    return (isSame(this->_yaw, other._yaw) &&
            isSame(this->_pitch, other._pitch) &&
            isSame(this->_roll, other._roll));
}

Quaternion EulerAngles::toQuaternion() const { return Quaternion(*this); }

Vec3 EulerAngles::rotate(const Eigen::Vector3d& vec) const
{
    return Quaternion(*this).rotate(vec);
}

}} // namespace isce3::core
