//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#ifndef ISCE_CORE_ATTITUDE_H
#define ISCE_CORE_ATTITUDE_H

#include <string>
#include <vector>
#include <array>
#include "DateTime.h"
#include "Ellipsoid.h"

// Declarations
namespace isce {
    namespace core {
        // The attitude classes
        class Attitude;
        class Quaternion;
        class EulerAngles;
        // Enum for specifying attitude types
        enum AttitudeType {QUATERNION_T, EULERANGLES_T};
    }
}

// Parent Attitude class to be inherited
class isce::core::Attitude {

    public:
        // Basic constructor to set the attitude type string
        Attitude(AttitudeType atype) : _attitude_type(atype) {};

        // Virtual functions
        virtual std::vector<double> ypr() = 0;
        virtual std::vector<std::vector<double>> rotmat(const std::string) = 0;

        // Getter functions
        inline AttitudeType attitudeType() const {return _attitude_type;}
        inline std::string yawOrientation() const {return _yaw_orientation;}

        // Setter functions
        inline void yawOrientation(const std::string);

    // Private data members
    private:
        DateTime _time;
        AttitudeType _attitude_type;
        std::string _yaw_orientation;
        
};

// Go ahead and define setYawOrientation here
void isce::core::Attitude::yawOrientation(const std::string orientation) {
    _yaw_orientation = orientation;
}

// Quaternion representation of attitude
class isce::core::Quaternion : public isce::core::Attitude {

    public:
        // Constructors
        Quaternion();
        Quaternion(std::vector<double> &);

        // Representations
        std::vector<double> ypr();
        std::vector<std::vector<double>> rotmat(const std::string);
        std::vector<double> factoredYPR(std::vector<double> &,
            std::vector<double> &, Ellipsoid *);

        // Get a copy of the quaternion elements
        inline std::vector<double> qvec() const;
        // Set all quaternion elements from a vector
        inline void qvec(const std::vector<double> &);

        // Get an individual quaternion element
        inline double qvecElement(const int) const;
        // Set individual quaternion element
        inline void qvecElement(const double, const int);
        
    // Private data members
    private:
        std::vector<double> _qvec;
};

// Euler angle attitude representation
class isce::core::EulerAngles : public isce::core::Attitude {

    public:
        // Constructors
        EulerAngles(double yaw=0.0, double pitch=0.0, double roll=0.0,
            const std::string yaw_orientation="normal");

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

        // Get the attitude angles
        inline double yaw() const;
        inline double pitch() const;
        inline double roll() const;

        // Set the attitude angles
        inline void yaw(const double);
        inline void pitch(const double);
        inline void roll(const double);

    // Private data members
    private:
        // Attitude angles
        double _yaw, _pitch, _roll;

};

// Get inline implementations for Quaternion
#define ISCE_CORE_QUATERNION_ICC
#include "Quaternion.icc"
#undef ISCE_CORE_QUATERNION_ICC

// Get inline implementations for EulerAngles
#define ISCE_CORE_EULERANGLES_ICC
#include "EulerAngles.icc"
#undef ISCE_CORE_EULERANGLES_ICC

#endif

// end of file
