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
#include "Constants.h"
#include "DateTime.h"

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
        Attitude(AttitudeType atype) : _time(MIN_DATE_TIME), _attitude_type(atype) {};

        // Virtual functions
        virtual cartesian_t ypr() = 0;
        virtual cartmat_t rotmat(const std::string) = 0;

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

#endif

// end of file
