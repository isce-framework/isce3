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
        EulerAngles(const std::string yaw_orientation="normal");

        /** Constructor with vectors of time and attitude angles */
        EulerAngles(const std::vector<double> & time,
                    const std::vector<double> & yaw,
                    const std::vector<double> & pitch,
                    const std::vector<double> & roll,
                    const std::string yaw_orientation="normal");

        /** Copy constructor */
        EulerAngles(const EulerAngles &);

        /** Comparison operator */
        bool operator==(const EulerAngles &) const;

        /** Assignment operator */
        EulerAngles & operator=(const EulerAngles &);

        /** Set data after construction */
        void data(const std::vector<double> & time,
                  const std::vector<double> & yaw,
                  const std::vector<double> & pitch,
                  const std::vector<double> & roll);

        /** Return data vector of time */
        inline const std::vector<double> & time() const { return _time; }

        /** Return data vector of yaw */
        inline const std::vector<double> & yaw() const { return _yaw; }

        /** Return data vector of pitch */
        inline const std::vector<double> & pitch() const { return _pitch; }

        /** Return data vector of roll */
        inline const std::vector<double> & roll() const { return _roll; }
    
        /** Interpolate yaw, pitch and roll at a given time */
        void ypr(double tintp, double & yaw, double & pitch, double & roll);

        /** Return rotation matrix at a given time with optional angle perturbations */
        cartmat_t rotmat(double tintp, const std::string,
                         double dyaw = 0.0, double dpitch = 0.0,
                         double d2 = 0.0, double d3 = 0.0);

        /** Return equivalent quaternion elements at a given time */
        std::vector<double> toQuaternionElements(double tintp);

        /** Return equivalent quaternion data structure */
        Quaternion toQuaternion();

        /** Return T3 rotation matrix around Z-axis */
        cartmat_t T3(double);

        /** Return T2 rotation matrix around Y-axis */
        cartmat_t T2(double);

        /** Return T1 rotation matrix around X-axis*/
        cartmat_t T1(double);

        /** Utility method to convert rotation matrix to Euler angles */
        static cartesian_t rotmat2ypr(const cartmat_t &);

        /** Get reference epoch */
        inline const isce::core::DateTime & refEpoch() const { return _refEpoch; }
        /** Set reference epoch */
        inline void refEpoch(const isce::core::DateTime & epoch) { _refEpoch = epoch; }

        /** Return number of epochs */
        inline size_t nVectors() const { return _yaw.size(); }

    // Private data members
    private:
        // Vectors of time and attitude angles
        std::vector<double> _time;
        std::vector<double> _yaw;
        std::vector<double> _pitch;
        std::vector<double> _roll;

        // Reference epoch
        isce::core::DateTime _refEpoch;
};

#endif

// end of file
