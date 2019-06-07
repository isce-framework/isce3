//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#ifndef ISCE_CORE_QUATERNION_H
#define ISCE_CORE_QUATERNION_H
#pragma once

#include "forward.h"

#include "Attitude.h"

/** Quaternion representation of attitude information*/
class isce::core::Quaternion : public isce::core::Attitude {

    public:
        /**Default constructor*/
        Quaternion();

        /** Constructor using vectors of time and quaternions */
        Quaternion(const std::vector<double> & time, const std::vector<double> & quaternions);

        /** Set all quaternion elements from a vector*/
        inline void data(const std::vector<double> & time,
                         const std::vector<double> & quaternions);

        /** Return yaw, pitch, roll*/
        void ypr(double tintp, double & yaw, double & pitch, double & roll);

        /** Return rotation matrix with optional perturbations */
        cartmat_t rotmat(double tintp, const std::string, double dq0 = 0.0, double dq1 = 0.0,
                         double dq2 = 0.0, double dq3 = 0.0);

        /** Return factored Yaw, Pitch, Roll */
        cartesian_t factoredYPR(double time,
                                const cartesian_t &,
                                const cartesian_t &,
                                Ellipsoid *);

        /** Get a copy of the quaternion elements*/
        inline std::vector<double> qvec() const { return _qvec; };

        /** Return number of epochs */
        inline size_t nVectors() const { return _time.size(); }

    // Private data members
    private:
        std::vector<double> _time;
        std::vector<double> _qvec;
};

#endif

// end of file
