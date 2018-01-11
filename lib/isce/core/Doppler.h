//-*- C++ -*-
//-*- coding: utf-8 -*-

#ifndef DOPPLER_H
#define DOPPLER_H

#include <string>
#include "Orbit.h"
#include "Pegtrans.h"
#include "Ellipsoid.h"
#include "Attitude.h"

namespace isce {
namespace core {

template<class Attitude>
struct Doppler {

    // Structs
    Ellipsoid * ellipsoid;
    Orbit * orbit;
    Attitude * attitude;
    Pegtrans ptm;

    // State vectors
    vector_t satxyz;
    vector_t satvel;
    vector_t satllh;
    double epoch;

    // Constructors
    Doppler() {};
    Doppler(Orbit *, Attitude *, Ellipsoid *, double);

    double centroid(double, double, std::string, size_t max_iter=10, int side=-1,
        bool precession=false);

};

template struct Doppler<EulerAngles>;
template struct Doppler<Quaternion>;

} // namespace core
} // namespace isce

#endif

// end of file
