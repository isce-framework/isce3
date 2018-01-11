//-*- C++ -*-
//-*- coding: utf-8 -*-

#ifndef ATTITUDE_H
#define ATTITUDE_H

#include <string>
#include <vector>
#include <array>
#include "DateTime.h"
#include "Ellipsoid.h"

namespace isce {
namespace core {

// Pure abstract Attitude class to be inherited
struct Attitude {
    // Basic constructor to set the attitude type string
    std::string attitude_type;
    Attitude(std::string type_name) : attitude_type(type_name) {};
    // Virtual functions
    virtual std::vector<double> ypr() = 0;
    virtual std::vector<std::vector<double>> rotmat(const std::string) = 0;
};

// Quaternion representation of attitude
struct Quaternion : public Attitude {

    // Attributes
    DateTime time;
    std::vector<double> qvec;

    // Constructor
    Quaternion();
    Quaternion(std::vector<double> &);

    // Representations
    std::vector<double> ypr();
    std::vector<std::vector<double>> rotmat(const std::string);
    std::vector<double> factoredYPR(std::vector<double> &, std::vector<double> &, Ellipsoid *);
};

// Euler angle attitude representation
struct EulerAngles : public Attitude {

    // Attributes
    DateTime time;
    double yaw, pitch, roll;

    // Constructor
    EulerAngles(double yaw=0.0, double pitch=0.0, double roll=0.0, bool deg=false);

    // Representations
    std::vector<double> ypr();
    std::vector<std::vector<double>> rotmat(const std::string);
    std::vector<double> toQuaternionElements();
    Quaternion toQuaternion();

    // Elementary rotation matrices
    std::vector<std::vector<double>> T3(double);
    std::vector<std::vector<double>> T2(double);
    std::vector<std::vector<double>> T1(double);

    // Utility method to convert rotation matrix to Euler angles
    static std::vector<double> rotmat2ypr(std::vector<std::vector<double>> &);
};

} // namespace core
} // namespace isce

#endif

// end of file
