#pragma once
#define EIGEN_MPL2_ONLY

#include "forward.h"

#include <Eigen/Geometry>

#include "DenseMatrix.h"

namespace isce3 { namespace core {

/** Quaternion representation of rotations, based on Eigen::Quaternion.
 *
 * Element names correspond to q = w + xi + yj + zk.
 * Uses Hamilton convention, same as SPICE toolikt
 * (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/qxq_c.html).
 * See https://arxiv.org/pdf/1801.07478.pdf for details of conventions.
 */
class Quaternion : public Eigen::Quaternion<double> {
    using super_t = Eigen::Quaternion<double>;
public:
    // Don't inherit constructors since array ctor uses different order than our
    // serialization. Should arguably hide super_t::coeffs() for the same
    // reason, but don't want to break inheritance relationship over one method.
    // Also don't inherit ctors so we can enforce normalization since we only
    // want to represent rotations.

    /** Constructor.  Normalizes elements */
    Quaternion(double w, double x, double y, double z) : super_t(w, x, y, z)
    {
        normalize();
    }

    /** Copy and normalize. */
    Quaternion(const Eigen::Quaternion<double>& other) : super_t(other)
    {
        normalize();
    }

    /** Convert from (yaw, pitch, roll) */
    Quaternion(const EulerAngles& ypr);

    /** Convert from rotation matrix. */
    explicit Quaternion(const Mat3& rotmat) : super_t(rotmat) {}

    /** Default constructor, pure real/no rotation. */
    Quaternion() : super_t(1, 0, 0, 0) {}
};

}} // namespace isce3::core
