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
#include "Attitude.h"

// EulerAngle constructor
isce::core::EulerAngles::
EulerAngles(double yaw, double pitch, double roll, const std::string yaw_orientation) 
    : Attitude("euler") {
    this->yaw = yaw;
    this->pitch = pitch;
    this->roll = roll;
    if (yaw_orientation.compare("normal") == 0 || yaw_orientation.compare("center") == 0) {
        this->yaw_orientation = yaw_orientation;
    } else {
        std::cerr << "Unsupported yaw orientation. Must be normal or center." << std::endl;
        throw std::invalid_argument("Unsupported yaw orientation.");
    }
}

// Return vector of Euler angles
std::vector<double>
isce::core::EulerAngles::ypr() {
    std::vector<double> v = {yaw, pitch, roll};
    return v;
}

// Rotation matrix for a given sequence
std::vector<std::vector<double>>
isce::core::EulerAngles::rotmat(const std::string sequence) {
    
    // Construct map for Euler angles to elementary rotation matrices
    std::map<const char, std::vector<std::vector<double>>> R_map;
    R_map['y'] = T3(yaw);
    R_map['p'] = T2(pitch);
    R_map['r'] = T1(roll);

    // Build composite matrix
    std::vector<std::vector<double>> R(3, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> R_tmp(3, std::vector<double>(3, 0.0));
    LinAlg::matMat(R_map[sequence[1]], R_map[sequence[2]], R_tmp);
    LinAlg::matMat(R_map[sequence[0]], R_tmp, R);

    return R;

}

// Rotation around Z-axis
std::vector<std::vector<double>>
isce::core::EulerAngles::T3(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    std::vector<std::vector<double>> T{{
        {cos, -sin, 0.0},
        {sin, cos, 0.0},
        {0.0, 0.0, 1.0}
    }};
    return T;
}

// Rotation around Y-axis
std::vector<std::vector<double>>
isce::core::EulerAngles::T2(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    std::vector<std::vector<double>> T{{
        {cos, 0.0, sin},
        {0.0, 1.0, 0.0},
        {-sin, 0.0, cos}
    }};
    return T;
}

// Rotation around X-axis
std::vector<std::vector<double>>
isce::core::EulerAngles::T1(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    std::vector<std::vector<double>> T{{
        {1.0, 0.0, 0.0},
        {0.0, cos, -sin},
        {0.0, sin, cos}
    }};
    return T;
}

// Extract YPR angles from a rotation matrix
std::vector<double>
isce::core::EulerAngles::rotmat2ypr(std::vector<std::vector<double>> & R) {

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
    std::vector<double> angles{yaw, pitch, roll};
    return angles;
}

// Return quaternion elements; multiply by -1 to get consistent signs
std::vector<double>
isce::core::EulerAngles::toQuaternionElements() {
    // Compute trig quantities
    const double cy = std::cos(yaw * 0.5);
    const double sy = std::sin(yaw * 0.5);
    const double cp = std::cos(pitch * 0.5);
    const double sp = std::sin(pitch * 0.5);
    const double cr = std::cos(roll * 0.5);
    const double sr = std::sin(roll * 0.5);
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

// Quaternion default constructor
isce::core::Quaternion::
Quaternion() : Attitude("quaternion"), qvec{0.0, 0.0, 0.0, 0.0} {}
// Quaternion constructor with vector
isce::core::Quaternion::
Quaternion(std::vector<double> & q) : Attitude("quaternion"), qvec(q) {}

// Return vector of Euler angles
std::vector<double>
isce::core::Quaternion::ypr() {

    // Get quaternion elements
    double q0 = qvec[0];
    double q1 = qvec[1];
    double q2 = qvec[2];
    double q3 = qvec[3];

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
    const double yaw = std::atan2(r11, r12);
    const double pitch = std::asin(r21);
    const double roll = std::atan2(r31, r32);

    // Make vector and return
    std::vector<double> angles{yaw, pitch, roll};
    return angles;
    
}


// Convert quaternion to rotation matrix
std::vector<std::vector<double>>
isce::core::Quaternion::rotmat(const std::string dummy) {

    // Cache quaternion elements
    if (qvec.size() != 4) {
        std::cerr << "ERROR: quaternion does not have the right size" << std::endl;
    }
    const double a = qvec[0];
    const double b = qvec[1];
    const double c = qvec[2];
    const double d = qvec[3];

    // Construct rotation matrix
    std::vector<std::vector<double>> R{{
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

// Extract YPR after factoring out orbit matrix
std::vector<double>
isce::core::Quaternion::factoredYPR(std::vector<double> & satxyz, std::vector<double> & satvel,
    Ellipsoid * ellipsoid) {

    // Compute ECI velocity assuming attitude angles are provided in inertial frame
    std::vector<double> Va(3);
    std::vector<double> w{0.0, 0.0, 0.00007292115833};
    LinAlg::cross(w, satxyz, Va);
    for (size_t i = 0; i < 3; ++i)
        Va[i] += satvel[i];

    // Compute vectors for TCN-like basis
    std::vector<double> q(3), c(3), b(3), a(3), temp(3);
    temp = {satxyz[0], satxyz[1], satxyz[2] / (1 - ellipsoid->e2)};
    LinAlg::unitVec(temp, q);
    c = {-q[0], -q[1], -q[2]};
    LinAlg::cross(c, Va, temp);
    LinAlg::unitVec(temp, b);
    LinAlg::cross(b, c, a);

    // Stack basis vectors to get transposed orbit matrix
    std::vector<std::vector<double>> L0(3, std::vector<double>(3, 0.0));
    for (size_t i = 0; i < 3; ++i) {
        L0[0][i] = a[i];
        L0[1][i] = b[i];
        L0[2][i] = c[i];
    }

    // Get total rotation matrix
    std::vector<std::vector<double>> R = rotmat("");

    // Multiply by transpose to get pure attitude matrix
    std::vector<std::vector<double>> L(3, std::vector<double>(3, 0.0));
    LinAlg::matMat(L0, R, L);

    // Extract Euler angles from rotation matrix
    std::vector<double> angles = EulerAngles::rotmat2ypr(L);
    return angles;
}

// end of file
