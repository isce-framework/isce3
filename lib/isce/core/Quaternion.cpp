//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <string>
#include <cmath>
#include <map>

#include "LinAlg.h"
#include "Quaternion.h"
#include "EulerAngles.h"

// Quaternion default constructor
isce::core::Quaternion::
Quaternion() : Attitude(QUATERNION_T) {}

// Quaternion constructor with vectors of time and quaternions
isce::core::Quaternion::
Quaternion(const std::vector<double> & time, const std::vector<double> & quaternions) :
           Attitude(QUATERNION_T) {
    this->data(time, quaternions); 
}

// Return vector of Euler angles evaluated at a given time
/** @param[in] tintp Seconds since reference epoch 
  * @param[out] oyaw Interpolated yaw angle
  * @param[out] opitch Interpolated pitch angle
  * @param[out] oroll Interpolated roll angle */
void
isce::core::Quaternion::ypr(double tintp, double & yaw, double & pitch, double & roll) {

    // Check time bounds; warn if invalid time requested
    const int n = nVectors();
    if (tintp < _time[0] || tintp > _time[n-1]) {
        pyre::journal::warning_t warnChannel("isce.core.Quaternion");
        warnChannel
            << pyre::journal::at(__HERE__)
            << "Requested out-of-bounds time. Attitude will be invalid."
            << pyre::journal::endl;
        return;
    }

    // For now, we only implement nearest neighbor
    int idx = -1;
    for (int i = 0; i < n; ++i) {
        if (_time[i] >= tintp) {
            idx = i;
            break;
        }
    }

    // Get quaternion elements
    double q0 = _qvec[idx*4 + 0];
    double q1 = _qvec[idx*4 + 1];
    double q2 = _qvec[idx*4 + 2];
    double q3 = _qvec[idx*4 + 3];

    // Compute quaternion norm
    const double qmod = std::sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    // Normalize elements
    q0 /= qmod;
    q1 /= qmod;
    q2 /= qmod;
    q3 /= qmod;

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
rotmat(double tintp, const std::string dummy, double dq0, double dq1, double dq2, double dq3) {

    // Check time bounds; error if out of bonds
    const int n = nVectors();
    if (tintp < _time[0] || tintp > _time[n-1]) {
        pyre::journal::error_t errorChannel("isce.core.Quaternion");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Requested out-of-bounds time."
            << pyre::journal::endl;
    }

    // For now, we only implement nearest neighbor
    int idx = -1;
    for (int i = 0; i < n; ++i) {
        if (_time[i] >= tintp) {
            idx = i;
            break;
        }
    }

    // Get quaternion elements
    const double a = _qvec[idx*4 + 0] + dq0;
    const double b = _qvec[idx*4 + 1] + dq1;
    const double c = _qvec[idx*4 + 2] + dq2;
    const double d = _qvec[idx*4 + 3] + dq2;

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
    cartesian_t Va;
    cartesian_t w{0.0, 0.0, 0.00007292115833};
    LinAlg::cross(w, satxyz, Va);
    for (size_t i = 0; i < 3; ++i)
        Va[i] += satvel[i];

    // Compute vectors for TCN-like basis
    cartesian_t q, c, b, a, temp;
    temp = {satxyz[0], satxyz[1], satxyz[2] / (1 - ellipsoid->e2())};
    LinAlg::unitVec(temp, q);
    c = {-q[0], -q[1], -q[2]};
    LinAlg::cross(c, Va, temp);
    LinAlg::unitVec(temp, b);
    LinAlg::cross(b, c, a);

    // Stack basis vectors to get transposed orbit matrix
    cartmat_t L0;
    for (size_t i = 0; i < 3; ++i) {
        L0[0][i] = a[i];
        L0[1][i] = b[i];
        L0[2][i] = c[i];
    }

    // Get total rotation matrix
    cartmat_t R = rotmat(tintp, "");

    // Multiply by transpose to get pure attitude matrix
    cartmat_t L;
    LinAlg::matMat(L0, R, L);

    // Extract Euler angles from rotation matrix
    cartesian_t angles = EulerAngles::rotmat2ypr(L);
    return angles;
}

// end of file
