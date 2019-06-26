//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

// std
#include <iostream>
#include <string>
#include <cmath>
#include <map>

// pyre
#include <pyre/journal.h>

// isce::core
#include "EulerAngles.h"
#include "Utilities.h"

/** @param[in] yaw_orientation Can be "normal" or "center" */
isce::core::EulerAngles::
EulerAngles(const std::string yaw_orientation) 
    : Attitude(EULERANGLES_T) {
    if (yaw_orientation.compare("normal") == 0 || yaw_orientation.compare("center") == 0) {
        yawOrientation(yaw_orientation);
    } else {
        std::cerr << "Unsupported yaw orientation. Must be normal or center." << std::endl;
        throw std::invalid_argument("Unsupported yaw orientation.");
    }
}

// Constructor with data vectors
/** @param[in] time Vector of observation times in seconds since reference epoch
  * @param[in] yaw Vector of yaw angles
  * @param[in] pitch Vector of pitch angles
  * @param[in] roll Vector of roll angles */
isce::core::EulerAngles::
EulerAngles(const std::vector<double> & time, const std::vector<double> & yaw,
            const std::vector<double> & pitch, const std::vector<double> & roll,
            const std::string yaw_orientation) : EulerAngles(yaw_orientation) {
    // Call setter for data
    this->data(time, yaw, pitch, roll);
}

// Copy constructor
/** @param[in] euler EulerAngles object */
isce::core::EulerAngles::
EulerAngles(const EulerAngles & euler) : Attitude(EULERANGLES_T),
                                         _time(euler.time()), _yaw(euler.yaw()),
                                         _pitch(euler.pitch()), _roll(euler.roll()) {
    const std::string yaw_orientation = euler.yawOrientation();
    if (yaw_orientation.compare("normal") == 0 || yaw_orientation.compare("center") == 0) {
        yawOrientation(yaw_orientation);
    } else {
        std::cerr << "Unsupported yaw orientation. Must be normal or center." << std::endl;
        throw std::invalid_argument("Unsupported yaw orientation.");
    }
}

// Comparison operator
bool isce::core::EulerAngles::
operator==(const EulerAngles & other) const {
    // Easy checks first
    bool equal = this->nVectors() == other.nVectors();
    equal *= _refEpoch == other.refEpoch();
    if (!equal) {
        return false;
    }
    // If we pass the easy checks, check the contents
    for (size_t i = 0; i < this->nVectors(); ++i) {
        equal *= isce::core::compareFloatingPoint(_time[i], other.time()[i]);
        equal *= isce::core::compareFloatingPoint(_yaw[i], other.yaw()[i]);
        equal *= isce::core::compareFloatingPoint(_pitch[i], other.pitch()[i]);
        equal *= isce::core::compareFloatingPoint(_roll[i], other.roll()[i]);
    }
    return equal;
}

// Assignment operator
/** @param[in] euler EulerAngles object */
isce::core::EulerAngles &
isce::core::EulerAngles::
operator=(const EulerAngles & euler) {
    _time = euler.time();
    _yaw = euler.yaw();
    _pitch = euler.pitch();
    _roll = euler.roll();
    _refEpoch = euler.refEpoch();
    yawOrientation(euler.yawOrientation());
    return *this;
}

// Set data after construction
/** @param[in] time Vector of observation times in seconds since reference epoch
  * @param[in] yaw Vector of yaw angles
  * @param[in] pitch Vector of pitch angles
  * @param[in] roll Vector of roll angles */
void
isce::core::EulerAngles::
data(const std::vector<double> & time, const std::vector<double> & yaw,
     const std::vector<double> & pitch, const std::vector<double> & roll) {
    // Check size consistency
    const bool flag = (time.size() == yaw.size()) && (yaw.size() == pitch.size()) &&
                      (pitch.size() == roll.size());
    if (!flag) {
        pyre::journal::error_t errorChannel("isce.core.EulerAngles");
        errorChannel    
            << pyre::journal::at(__HERE__)
            << "Inconsistent vector sizes"
            << pyre::journal::endl;
    }
    // Copy vectors
    _time = time;
    _yaw = yaw;
    _pitch = pitch;
    _roll = roll;
}
    
/** @param[in] tintp Seconds since reference epoch 
  * @param[out] oyaw Interpolated yaw angle
  * @param[out] opitch Interpolated pitch angle
  * @param[out] oroll Interpolated roll angle */
void
isce::core::EulerAngles::
ypr(double tintp, double & oyaw, double & opitch, double & oroll) {

    // Check we have enough state vectors
    const int n = nVectors();
    if (n < 9) {
        pyre::journal::error_t errorChannel("isce.core.EulerAngles");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "EulerAngles::ypr requires at least 9 state vectors to interpolate. "
            << "EulerAngles only contains " << n
            << pyre::journal::endl;
    }
    
    // Check time bounds; warn if invalid time requested
    if (tintp < _time[0] || tintp > _time[n-1]) {
        pyre::journal::warning_t warnChannel("isce.core.EulerAngles");
        warnChannel
            << pyre::journal::at(__HERE__)
            << "Requested out-of-bounds time. Attitude will be invalid."
            << pyre::journal::endl;
        return;
    }

    // Get nearest time index
    int idx = -1;
    for (int i = 0; i < n; ++i) {
        if (_time[i] >= tintp) {
            idx = i;
            break;
        }
    }
    idx -= 5;
    idx = std::min(std::max(idx, 0), n - 9);

    // Load inteprolation arrays
    std::vector<double> t(9), yaw(9), pitch(9), roll(9);
    for (int i = 0; i < 9; i++) {
        t[i] = _time[idx+i];
        yaw[i] = _yaw[idx+i];
        pitch[i] = _pitch[idx+i];
        roll[i] = _roll[idx+i];
    }

    const double trel = (8. * (tintp - t[0])) / (t[8] - t[0]);
    double teller = 1.0;
    for (int i = 0; i < 9; i++)
        teller *= trel - i;

    // Perform polynomial interpolation
    oyaw = 0.0;
    opitch = 0.0;
    oroll = 0.0;
    if (teller == 0.0) {
        oyaw = yaw[int(trel)];
        opitch = pitch[int(trel)];
        oroll = roll[int(trel)];
    } else {
        const std::vector<double> noemer = {
            40320.0, -5040.0, 1440.0, -720.0, 576.0, -720.0, 1440.0, -5040.0, 40320.0
        };
        for (int i = 0; i < 9; i++) {
            double coeff = (teller / noemer[i]) / (trel - i);
            oyaw += coeff * yaw[i];
            opitch += coeff * pitch[i];
            oroll += coeff * roll[i];
        }
    }
}

// Rotation matrix for a given sequence and time
/** @param[in] tintp Seconds since reference epoch
  * @param[in] sequence String of rotation sequence
  * @param[in] dyaw Yaw perturbation
  * @param[in] dpitch Pitch perturbation
  * @param[in] d2 (Not used)
  * @param[in] d3 (Not used)
  * @param[out] R Rotation matrix for given sequence */
isce::core::cartmat_t
isce::core::EulerAngles::
rotmat(double tintp, const std::string sequence, double dyaw, double dpitch,
       double, double) {

    // Interpolate to get YPR angles
    double yaw, pitch, roll;
    this->ypr(tintp, yaw, pitch, roll);

    // Construct map for Euler angles to elementary rotation matrices
    std::map<const char, cartmat_t> R_map;
    R_map['y'] = T3(yaw + dyaw);
    R_map['p'] = T2(pitch + dpitch);
    R_map['r'] = T1(roll);

    // Build composite matrix
    return R_map[sequence[0]].dot(
            R_map[sequence[1]].dot(R_map[sequence[2]])
           );
}

// Rotation around Z-axis
isce::core::cartmat_t
isce::core::EulerAngles::T3(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    return cartmat_t {{
        {cos, -sin, 0.0},
        {sin, cos, 0.0},
        {0.0, 0.0, 1.0}
    }};
}

// Rotation around Y-axis
isce::core::cartmat_t
isce::core::EulerAngles::T2(double angle) {
    const double cos = std::cos(angle);
    const double sin = std::sin(angle);
    return cartmat_t {{
        {cos, 0.0, sin},
        {0.0, 1.0, 0.0},
        {-sin, 0.0, cos}
    }};
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
isce::core::EulerAngles::rotmat2ypr(const cartmat_t & R) {

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

// Return quaternion elements at a given time; multiply by -1 to get consistent signs
/** @param[in] tintp Seconds since reference epoch
  * @param[out] q Vector of quaternion elements */
std::vector<double>
isce::core::EulerAngles::toQuaternionElements(double tintp) {

    // Interpolate to get YPR angles
    double yaw, pitch, roll;
    this->ypr(tintp, yaw, pitch, roll);

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

// Return quaternion representation for all epochs
/** @param[out] quat Quaternion instance */
isce::core::Quaternion
isce::core::EulerAngles::toQuaternion() {
    // Vector to fill
    std::vector<double> qvec(nVectors()*4);
    // Loop over epochs and convert to quaternion values
    for (size_t i = 0; i < nVectors(); ++i) {
        std::vector<double> q = toQuaternionElements(_time[i]);
        for (size_t j = 0; j < 4; ++j) {
            qvec[i*4 + j] = q[j];
        }
    }
    // Make a quaternion
    Quaternion quat(_time, qvec);
    return quat;
}

// end of file
