#include "Basis.h"

#include "Ellipsoid.h"
#include "EulerAngles.h"
#include "Quaternion.h"

namespace isce3 { namespace core {

Vec3 velocityECI(const Vec3& position, const Vec3& velocityECF)
{
    Vec3 omega {0.0, 0.0, isce3::core::EarthSpinRate};
    return velocityECF + omega.cross(position);
}

Vec3 velocityECF(const Vec3& position, const Vec3& velocityECI)
{
    Vec3 omega {0.0, 0.0, isce3::core::EarthSpinRate};
    return velocityECI - omega.cross(position);
}

Basis geodeticTCN(const Vec3& x, const Vec3& v, const Ellipsoid& ellipsoid)
{
    const Vec3 temp(x[0], x[1], x[2] / (1 - ellipsoid.e2()));
    return Basis(temp, v);
}

static EulerAngles ypr_helper(const Basis& tcn2xyz, const Quaternion& q)
{
    const Mat3 R = q.toRotationMatrix();
    const Mat3 L0 = tcn2xyz.toRotationMatrix();
    return EulerAngles(L0.transpose().dot(R));
}

EulerAngles factoredYawPitchRoll(const Quaternion& q, const Vec3& x,
                                 const Vec3& v, const Ellipsoid& ellipsoid)
{
    auto velECI = velocityECI(x, v);
    auto geodetic_tcn = geodeticTCN(x, velECI, ellipsoid);
    return ypr_helper(geodetic_tcn, q);
}

EulerAngles factoredYawPitchRoll(const Quaternion& q, const Vec3& x,
                                 const Vec3& v)
{
    auto velECI = velocityECI(x, v);
    auto geocentric_tcn = Basis(x, velECI);
    return ypr_helper(geocentric_tcn, q);
}

}} // namespace isce3::core
