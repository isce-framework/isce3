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

// Some useful typedefs
typedef std::vector<double> vector_t;
typedef std::vector<std::vector<double>> matrix_t;

// Pure abstract Attitude class to be inherited
struct BaseAttitude {
    virtual vector_t ypr() = 0;
    virtual matrix_t rotmat(const std::string) = 0;
};

// Quaternion representation of attitude
struct Quaternion : public BaseAttitude {

    // Attributes
    DateTime time;
    vector_t qvec;

    // Constructor
    Quaternion();
    Quaternion(vector_t &);

    // Representations
    vector_t ypr();
    matrix_t rotmat(const std::string);
    vector_t factoredYPR(vector_t &, vector_t &, Ellipsoid *);
};

// Euler angle attitude representation
struct EulerAngles : public BaseAttitude {

    // Attributes
    DateTime time;
    double yaw, pitch, roll;

    // Constructor
    EulerAngles(double yaw=0.0, double pitch=0.0, double roll=0.0, bool deg=false);

    // Representations
    vector_t ypr();
    matrix_t rotmat(const std::string);
    vector_t toQuaternionElements();
    Quaternion toQuaternion();

    // Elementary rotation matrices
    matrix_t T3(double);
    matrix_t T2(double);
    matrix_t T1(double);

    // Utility method to convert rotation matrix to Euler angles
    static vector_t rotmat2ypr(matrix_t &);
};

} // namespace core
} // namespace isce

#endif

// end of file
