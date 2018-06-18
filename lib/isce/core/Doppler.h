//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018

#ifndef ISCE_CORE_DOPPLER_H
#define ISCE_CORE_DOPPLER_H

#include <string>
#include "Constants.h"
#include "Orbit.h"
#include "Pegtrans.h"
#include "Ellipsoid.h"
#include "Attitude.h"

namespace isce {
namespace core {

/** Data structure for computing Doppler Centroids from Platform attitude and position*/
struct Doppler {

    // Structs
    Ellipsoid ellipsoid;
    Orbit orbit;
    Pegtrans ptm;
    // Pointer to base Attitude class
    Attitude * attitude;

    // State vectors
    cartesian_t satxyz;
    cartesian_t satvel;
    cartesian_t satllh;
    double epoch;

    /**Default empty constructor*/
    Doppler() {};

    /**Constructor with Orbit, Attitude and Ellipsoid objects*/
    Doppler(Orbit &orbit, Attitude *attitude, Ellipsoid &ellipsoid, double epoch);

    /**Compute the Doppler Centroid at given slant range*/
    double centroid(double slantRange, double wvl, std::string frame,
            size_t max_iter=10, int side=-1, bool precession=false);

};

} // namespace core
} // namespace isce

#endif

// end of file
