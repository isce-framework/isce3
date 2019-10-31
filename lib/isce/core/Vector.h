#pragma once

#include <array>
#include <cmath>
#define EIGEN_MPL2_ONLY
#include <Eigen/Dense>
#include "Common.h"

namespace isce { namespace core {

template<int N, typename T>
class Vector : public Eigen::Matrix<T, N, 1> {
    using super_t = Eigen::Matrix<T, N, 1>;
    using super_t::Matrix;
};

// Function to compute normal vector to a plane given three points
CUDA_HOSTDEV inline Vec3 normalPlane(const Vec3& p1,
                                     const Vec3& p2,
                                     const Vec3& p3) {
    const Vec3 p13 = p3 - p1;
    const Vec3 p12 = p2 - p1;
    return p13.cross(p12).normalized();
}

}} // namespace isce::core
