//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include "Quaternion.h"

#include <iostream>
#include <string>
#include <cmath>
#include <map>

#include <pyre/journal.h>

#include "DenseMatrix.h"
#include "Ellipsoid.h"
#include "EulerAngles.h"

constexpr auto Quaternion_t = isce::core::Attitude::Type::Quaternion_t;

// Quaternion default constructor
isce::core::Quaternion::
Quaternion() : Attitude(Quaternion_t) {}

// Quaternion constructor with vectors of time and quaternions
isce::core::Quaternion::
Quaternion(const std::vector<double> & time, const std::vector<double> & quaternions) :
           Attitude(Quaternion_t) {
    this->data(time, quaternions);
}


Eigen::Quaternion<double>
isce::core::Quaternion::_interp(double t) const
{
    // Check time bounds; error if out of bonds
    const int n = nVectors();
    if (t < _time[0] || t > _time[n-1]) {
        pyre::journal::error_t errorChannel("isce.core.Quaternion");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Requested out-of-bounds time."
            << pyre::journal::endl;
    }

    // Find interval containing desired point.
    // _time setter guarantees monotonic.
    // FIXME Use binary search or require equal time steps.
    int i;
    for (i = 1; i < n; ++i)
        if (_time[i] >= t)
            break;

    // Slerp between the nearest data points.
    const double tq = (t - _time[i-1]) / (_time[i] - _time[i-1]);
    Eigen::Quaternion q0(&_qvec[(i-1)*4]);
    Eigen::Quaternion q1(&_qvec[i*4]);
    return q0.slerp(tq, q1);
}


// Return vector of Euler angles evaluated at a given time
/** @param[in] tintp Seconds since reference epoch
  * @param[out] oyaw Interpolated yaw angle
  * @param[out] opitch Interpolated pitch angle
  * @param[out] oroll Interpolated roll angle */
void
isce::core::Quaternion::ypr(double tintp, double & yaw, double & pitch, double & roll)
{
    auto q = _interp(tintp);

    const double q0 = q.x(), q1 = q.y(), q2 = q.z(), q3 = q.w();

    // Compute Euler angles
    const double r11 = 2.0 * (q1*q2 + q0*q3);
    const double r12 = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    const double r21 = -2.0 * (q1*q3 - q0*q2);
    const double r31 = 2.0 * (q2*q3 + q0*q1);
    const double r32 = q0*q0 - q1*q1 - q2*q2 + q3*q3;
    yaw = std::atan2(r11, r12);
    pitch = std::asin(r21);
    roll = std::atan2(r31, r32);
}

// Convert quaternion to rotation matrix
isce::core::cartmat_t
isce::core::Quaternion::
rotmat(double tintp, const std::string, double dq0, double dq1, double dq2, double) {

    auto q = _interp(tintp);

    // Get quaternion elements
    const double
        a = q.x() + dq0,
        b = q.y() + dq1,
        c = q.z() + dq2,
        d = q.w() + dq2;  // WTF repeating dq2

    // Construct rotation matrix
    cartmat_t R{{
        {a*a + b*b - c*c - d*d,
         2*b*c - 2*a*d,
         2*b*d + 2*a*c},
        {2*b*c + 2*a*d,
         a*a - b*b + c*c - d*d,
         2*c*d - 2*a*b},
        {2*b*d - 2*a*c,
         2*c*d + 2*a*b,
         a*a - b*b - c*c + d*d}
    }};

    return R;
}

// Extract YPR at a given time after factoring out orbit matrix
isce::core::cartesian_t
isce::core::Quaternion::
factoredYPR(double tintp,
            const cartesian_t& satxyz,
            const cartesian_t& satvel,
            Ellipsoid * ellipsoid) {

    // Compute ECI velocity assuming attitude angles are provided in inertial frame
    cartesian_t w{0.0, 0.0, 0.00007292115833};
    const Vec3 Va = w.cross(satxyz) + satvel;

    // Compute vectors for TCN-like basis
    const Vec3 temp = {satxyz[0], satxyz[1], satxyz[2] / (1 - ellipsoid->e2())};
    const Vec3 c = -temp.normalized();
    const Vec3 b = c.cross(Va).normalized();
    const Vec3 a = b.cross(c);

    // Stack basis vectors to get transposed orbit matrix
    cartmat_t L0;
    for (size_t i = 0; i < 3; ++i) {
        L0[0][i] = a[i];
        L0[1][i] = b[i];
        L0[2][i] = c[i];
    }

    // Get total rotation matrix
    const cartmat_t R = rotmat(tintp, "");

    // Multiply by transpose to get pure attitude matrix
    const cartmat_t L = L0.dot(R);

    // Extract Euler angles from rotation matrix
    cartesian_t angles = EulerAngles::rotmat2ypr(L);
    return angles;
}

// Set quaternion elements from vectors
/** @param[in] time Vector of seconds since epoch, strictly increasing.
  * @param[in] quaternions Flattened vector of quaternions per time epoch */
void isce::core::Quaternion::data(const std::vector<double>& time,
                                  const std::vector<double>& quaternions) {
    // Check size consistency
    const bool flag = time.size() == (quaternions.size() / 4);
    if (!flag) {
        pyre::journal::error_t errorChannel("isce.core.Quaternion");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Inconsistent vector sizes"
            << pyre::journal::endl;
    }
    // Require strictly monotonic time stamps.
    for (int i=1; i<time.size(); ++i) {
        double dt = time[i] - time[i-1];
        if (dt <= 0.0) {
            pyre::journal::error_t errorChannel("isce.core.Quaternion");
            errorChannel << pyre::journal::at(__HERE__)
                << "Need strictly increasing time tags." << pyre::journal::endl;
        }
    }
    // Set data
    _time = time;
    _qvec = quaternions;
}

// end of file
