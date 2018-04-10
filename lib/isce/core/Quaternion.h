//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#ifndef ISCE_CORE_QUATERNION_H
#define ISCE_CORE_QUATERNION_H

#include "Attitude.h"
#include "Ellipsoid.h"

// Quaternion representation of attitude
class isce::core::Quaternion : public isce::core::Attitude {

    public:
        // Constructors
        Quaternion();
        Quaternion(std::vector<double> &);

        // Representations
        cartesian_t ypr();
        cartmat_t rotmat(const std::string);
        cartesian_t factoredYPR(const cartesian_t &, const cartesian_t &, Ellipsoid *);

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

// Get inline implementations for Quaternion
#define ISCE_CORE_QUATERNION_ICC
#include "Quaternion.icc"
#undef ISCE_CORE_QUATERNION_ICC

#endif

// end of file
