#include "Quaternion.h"

namespace isce3 { namespace core {

Quaternion::Quaternion(const EulerAngles& ypr)
    : Quaternion(ypr.yaw(), ypr.pitch(), ypr.roll())
{}

Vec3 Quaternion::toYPR() const
{
    double t0 {2.0 * (w() * x() + y() * z())};
    double t1 {1.0 - 2.0 * (x() * x() + y() * y())};
    double t2 {2.0 * (w() * y() - z() * x())};
    if (t2 > 1.0 || t2 < -1.0)
        t2 = 1.0;
    double t3 {2.0 * (w() * z() + x() * y())};
    double t4 {1.0 - 2.0 * (y() * y() + z() * z())};
    return Vec3(std::atan2(t3, t4), std::asin(t2), std::atan2(t0, t1));
}

EulerAngles Quaternion::toEulerAngles() const
{
    auto ypr {this->toYPR()};
    return EulerAngles(ypr(0), ypr(1), ypr(2));
}

}} // namespace isce3::core
