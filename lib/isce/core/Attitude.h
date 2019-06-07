//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#ifndef ISCE_CORE_ATTITUDE_H
#define ISCE_CORE_ATTITUDE_H
#pragma once

#include "forward.h"

#include <string>
#include <vector>
#include "Constants.h"
#include "DateTime.h"

/** Base class for attitude data representation */
class isce::core::Attitude {

    public:
        enum class Type { EulerAngles_t, Quaternion_t };

        /** Constructor using time attitude representation type*/
        Attitude(Attitude::Type atype) : _attitude_type(atype) {};

        /** Virtual destructor*/
        virtual ~Attitude() {}

        /** Virtual function to return yaw, pitch, roll */
        virtual void ypr(double tintp, double& yaw, double& pitch, double& roll) = 0;

        /** Virtual function return rotation matrix with optional perturbations */
        virtual Mat3 rotmat(double tintp, const std::string, double d0 = 0,
                            double d1 = 0, double d2 = 0, double d3 = 0) = 0;

        /** Return type of attitude representation - quaternion or euler angle*/
        inline Attitude::Type attitudeType()  const { return _attitude_type; }

        /** Return yaw orientation - central or normal */
        inline std::string yawOrientation() const { return _yaw_orientation; }

        /** Set yaw orientation - central or normal */
        inline void yawOrientation(const std::string);

    // Private data members
    private:
        Attitude::Type _attitude_type;
        std::string _yaw_orientation;
};

// Go ahead and define setYawOrientation here
void isce::core::Attitude::yawOrientation(const std::string orientation) {
    _yaw_orientation = orientation;
}

#endif

// end of file
