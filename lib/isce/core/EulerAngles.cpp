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
#include "EulerAngles.h"

// EulerAngle constructor
isce::core::EulerAngles::
EulerAngles(double yaw, double pitch, double roll, const std::string yaw_orientation) 
    : Attitude(EULERANGLES_T) {
    _yaw = yaw;
    _pitch = pitch;
    _roll = roll;
    if (yaw_orientation.compare("normal") == 0 || yaw_orientation.compare("center") == 0) {
        yawOrientation(yaw_orientation);
    } else {
        std::cerr << "Unsupported yaw orientation. Must be normal or center." << std::endl;
        throw std::invalid_argument("Unsupported yaw orientation.");
    }
}

// Return vector of Euler angles
isce::core::cartesian_t
isce::core::EulerAngles::ypr() {
    cartesian_t v = {_yaw, _pitch, _roll};
    return v;
}

// Rotation matrix for a given sequence
isce::core::cartmat_t
isce::core::EulerAngles::rotmat(const std::string sequence) {
    
    // Construct map for Euler angles to elementary rotation matrices
    std::map<const char, cartmat_t> R_map;
    R_map['y'] = T3(_yaw);
    R_map['p'] = T2(_pitch);
    R_map['r'] = T1(_roll);

    // Build composite matrix
    cartmat_t R, R_tmp;
    LinAlg::matMat(R_map[sequence[1]], R_map[sequence[2]], R_tmp);
    LinAlg::matMat(R_map[sequence[0]], R_tmp, R);

    return R;

}

// Rotation around Z-axis
isce::core::cartmat_t
isce::core::EulerAngles::T3(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    cartmat_t T{{
        {cos, -sin, 0.0},
        {sin, cos, 0.0},
        {0.0, 0.0, 1.0}
    }};
    return T;
}

// Rotation around Y-axis
isce::core::cartmat_t
isce::core::EulerAngles::T2(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    cartmat_t T{{
        {cos, 0.0, sin},
        {0.0, 1.0, 0.0},
        {-sin, 0.0, cos}
    }};
    return T;
}

// Rotation around X-axis
isce::core::cartmat_t
isce::core::EulerAngles::T1(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    cartmat_t T{{
        {1.0, 0.0, 0.0},
        {0.0, cos, -sin},
        {0.0, sin, cos}
    }};
    return T;
}

// Extract YPR angles from a rotation matrix
isce::core::cartesian_t
isce::core::EulerAngles::rotmat2ypr(cartmat_t & R) {

    const double sy = std::sqrt(R[0][0]*R[0][0] + R[1][0]*R[1][0]);
    double yaw, pitch, roll;
    if (sy >= 1.0e-6) {
        roll = std::atan2(R[2][1], R[2][2]);
        pitch = std::atan2(-R[2][0], sy);
        yaw = std::atan2(R[1][0], R[0][0]);
    } else {
        roll = std::atan2(-R[1][2], R[1][1]);
        pitch = std::atan2(-R[2][0], sy);
        yaw = 0.0;
    }

    // Make vector and return
    cartesian_t angles{yaw, pitch, roll};
    return angles;
}

// Return quaternion elements; multiply by -1 to get consistent signs
std::vector<double>
isce::core::EulerAngles::toQuaternionElements() {
    // Compute trig quantities
    const double cy = std::cos(_yaw * 0.5);
    const double sy = std::sin(_yaw * 0.5);
    const double cp = std::cos(_pitch * 0.5);
    const double sp = std::sin(_pitch * 0.5);
    const double cr = std::cos(_roll * 0.5);
    const double sr = std::sin(_roll * 0.5);
    // Make a vector
    std::vector<double> q = {
        -1.0 * (cy * cr * cp + sy * sr * sp),
        -1.0 * (cy * sr * cp - sy * cr * sp),
        -1.0 * (cy * cr * sp + sy * sr * cp),
        -1.0 * (sy * cr * cp - cy * sr * sp)
    };
    return q;
}

// Return quaternion representation
isce::core::Quaternion
isce::core::EulerAngles::toQuaternion() {
    // Get elements
    std::vector<double> qvec = toQuaternionElements();
    // Make a quaternion
    Quaternion quat(qvec);
    return quat;
}

// end of file
