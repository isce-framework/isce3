//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018

#ifndef ISCE_CORE_DOPPLER_H
#define ISCE_CORE_DOPPLER_H

#include <string>
#include "Orbit.h"
#include "Pegtrans.h"
#include "Ellipsoid.h"
#include "Attitude.h"

namespace isce {
namespace core {

struct Doppler {

    // Structs
    Ellipsoid ellipsoid;
    Orbit orbit;
    Pegtrans ptm;
    // Pointer to base Attitude class
    Attitude * attitude;

    // State vectors
    std::vector<double> satxyz;
    std::vector<double> satvel;
    std::vector<double> satllh;
    double epoch;

    // Constructors
    Doppler() {};
    Doppler(Orbit &, Attitude *, Ellipsoid &, double);

    double centroid(double, double, std::string, size_t max_iter=10, int side=-1,
        bool precession=false);

};

} // namespace core
} // namespace isce

#endif

// end of file
