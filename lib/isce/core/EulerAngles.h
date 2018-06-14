//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#ifndef ISCE_CORE_EULERANGLES_H
#define ISCE_CORE_EULERANGLES_H

#include "Attitude.h"
#include "Quaternion.h"

/** Data structure for Euler Angle representation of attitude information
 *
 * All angles are stored and returned in radians*/
class isce::core::EulerAngles : public isce::core::Attitude {

    public:
        /** Default constructor*/
        EulerAngles(double yaw=0.0, double pitch=0.0, double roll=0.0,
            const std::string yaw_orientation="normal");

        /** Return yaw, pitch and roll in a triplet*/
        cartesian_t ypr();

        /** Return rotation matrix*/
        cartmat_t rotmat(const std::string);

        /** Return equivalent quaternion elements*/
        std::vector<double> toQuaternionElements();

        /** Return equivalent quaternion data structure*/
        Quaternion toQuaternion();

        /** Return T3 rotation matrix around Z-axis*/
        cartmat_t T3(double);

        /** Return T2 rotation matrix around Y-axis*/
        cartmat_t T2(double);

        /** Return T1 rotation matrix around X-axis*/
        cartmat_t T1(double);

        /** Utility method to convert rotation matrix to Euler angles*/
        static cartesian_t rotmat2ypr(const cartmat_t &);

        /** Return yaw*/
        inline double yaw() const;

        /** Return pitch*/
        inline double pitch() const;

        /** Return roll */
        inline double roll() const;

        /** Set yaw */
        inline void yaw(const double);

        /** Set pitch */
        inline void pitch(const double);

        /** Set roll */
        inline void roll(const double);

    // Private data members
    private:
        // Attitude angles
        double _yaw, _pitch, _roll;

};

// Get inline implementations for EulerAngles
#define ISCE_CORE_EULERANGLES_ICC
#include "EulerAngles.icc"
#undef ISCE_CORE_EULERANGLES_ICC

#endif

// end of file
