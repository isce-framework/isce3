#pragma once
#define EIGEN_MPL2_ONLY

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <isce3/except/Error.h>

#include "DenseMatrix.h"
#include "EulerAngles.h"
#include "Vector.h"

namespace isce3 { namespace core {

/** Quaternion representation of rotations, based on double precision
 * Eigen::Quaterniond.
 *
 * Element names correspond to q = w + xi + yj + zk.
 * Uses Hamilton convention, same as SPICE toolikt
 * (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html).
 * See https://arxiv.org/pdf/1801.07478.pdf for details of conventions.
 */
class Quaternion : public Eigen::Quaterniond {

private:
    using super_t = Eigen::Quaterniond;
    using AngleAxis_t = Eigen::AngleAxisd;

public:
    /**
     * Default constructor
     * (qw,qx,qy,qz) = (1,0,0,0) , no rotation,
     * equivalent to 3x3 identity rotation matrix.
     */
    Quaternion() : super_t(1, 0, 0, 0) {}

    /**
     * Constructor from qw, qx, qy, qz
     * @param[in] w : double scalar , real part
     * @param[in] x : double scalar, imag part along x axis
     * @param[in] y : double scalar, imag part along y axis
     * @param[in] z : double scalar, imag part along z axis
     */
    Quaternion(double w, double x, double y, double z) : super_t(w, x, y, z)
    {
        normalize();
    }

    /**
     * Constructor from Eigen 4-element unit quaternion vector
     * @param[in] qvec : quaternion 4-element Eigen vector or
     * isce3 Vec4 representing qw, qx, qy, qz.
     */
    explicit Quaternion(const Eigen::Vector4d& qvec)
        : Quaternion(qvec(0), qvec(1), qvec(2), qvec(3))
    {}

    /**
     * Constructor from Eigen 3D unit vector
     * @param[in] vec : 3-D . 3-element Eigen vector
     * or isce3 Vec3 representing  qx, qy, qz.
     */
    explicit Quaternion(const Eigen::Vector3d& vec)
        : Quaternion(0.0, vec(0), vec(1), vec(2))
    {}

    /**
     * Constructor from an Eigen 3D Rotation matrix
     * Throw exception for non-uniary matrix within
     * 1e-6 precision.
     * @param[in] rotmat  : 3-by-3 rotation Eigen matrix
     * or isce3 Mat3.
     * @exception InvalidArgument
     */
    Quaternion(const Eigen::Matrix3d& rotmat) : super_t(rotmat)
    {
        if (!rotmat.isUnitary(1e-6))
            throw isce3::except::InvalidArgument(
                    ISCE_SRCINFO(), "Requires unitary/rotation Matrix!");
    }

    /**
     * Constructor from an Eigen AngleAxis object
     * @param[in] aa :  Eigen AngleAxisd object
     */
    Quaternion(const Eigen::AngleAxisd& aa)
        : super_t(AngleAxis_t(aa.angle(), aa.axis().normalized()))
    {}

    /**
     * Constructor from an angle and 3-D vector axis
     * @param[in] angle : scalar in radians
     * @param[in] axis : Eigen 3-D double unit vector
     */
    Quaternion(double angle, const Eigen::Vector3d& axis)
        : Quaternion(AngleAxis_t(angle, axis))
    {}

    /**
     * Constructor from Euler Angles yaw, pitch, roll
     * @param[in] yaw : scalar in radians
     * @param[in] pitch : scalar in radians
     * @param[in] roll : scalar in radians
     */
    Quaternion(double yaw, double pitch, double roll)
        : super_t((AngleAxis_t(yaw, Vec3::UnitZ()) *
                   AngleAxis_t(pitch, Vec3::UnitY()) *
                   AngleAxis_t(roll, Vec3::UnitX()))
                          .toRotationMatrix())
    {}

    /**
     * Constructor from isce3 EulerAngle Object
     * @param[in] ypr : isce3 EulerAngle object
     */
    explicit Quaternion(const EulerAngles& ypr);

    /**
     * Copy constructor from other Quaternion obj
     * @param[in] other : another Eigen quaternion object
     */
    Quaternion(const Eigen::Quaternion<double>& other) : super_t(other)
    {
        normalize();
    }

    /**
     * Rotate a 3-D vector by self quaternion object
     * @param[in] vec : Eigen Vector3 or isce3 Vec3
     * @return rotated Eigen Vector3 or isce3 Vec3
     */
    Eigen::Vector3d rotate(const Eigen::Vector3d& vec) const
    {
        // Note in case method "_tansformVector" of Eigen Base Quaternion
        // becomes private or deprecated, one can replace it with the
        // following code snippet :
        // return ((*this)*(Quaternion(vec)*(this->conjugate()))).vec();
        return this->_transformVector(vec);
    }

    /**
     * Convert quaternion to an Eigen vector of Euler angles
     * (yaw, pitch, roll) , all in radians
     * @return Eigen:Vector3 or isce3 Vec3 of
     * Yaw,Pitch,Roll angles all in radians.
     */
    Vec3 toYPR() const;

    /**
     * Build isce3 EulerAngles Object from Quaternion object
     * @return isce3 EulerAngle object
     */
    EulerAngles toEulerAngles() const;

    /**
     * Convert from Eigen Quaternion to Eigen AngleAxis object
     * @return Eigen AngleAxis object
     */
    Eigen::AngleAxisd toAngleAxis() const
    {
        double angh {std::acos(this->w())};
        return AngleAxis_t(2.0 * angh, (1.0 / std::sin(angh)) * (this->vec()));
    }
};

}} // namespace isce3::core
