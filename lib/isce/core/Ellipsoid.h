// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_ELLIPSOID_H
#define ISCE_CORE_ELLIPSOID_H

#include <cmath>
#include <array>
#include "Constants.h"

// Declaration
namespace isce {
    namespace core {
        class Ellipsoid;
    }
}

// Ellipsoid declaration
class isce::core::Ellipsoid {

    public:
        // Constructors
        Ellipsoid(double maj, double ecc) : _a(maj), _e2(ecc) {}
        Ellipsoid() : Ellipsoid(0.0, 0.0) {}
        Ellipsoid(const Ellipsoid & ellps) : _a(ellps.a()), _e2(ellps.e2()) {}
        inline Ellipsoid& operator=(const Ellipsoid&);

        // Get ellipsoid properties
        double a() const {return _a;}
        double e2() const {return _e2;}
        // Set ellipsoid properties
        void a(double val) {_a = val;}
        void e2(double val) {_e2 = val;}

        // Get radii in different directions
        inline double rEast(double) const;
        inline double rNorth(double) const;
        inline double rDir(double,double) const;

        // Transformation routines
        void latLonToXyz(const cartesian_t &, cartesian_t &) const;
        void xyzToLatLon(const cartesian_t &, cartesian_t &) const;
        void getAngs(const cartesian_t &,const cartesian_t &,
                     const cartesian_t &, double &, double &) const;
        void getTCN_TCvec(const cartesian_t &,const cartesian_t &,
                          const cartesian_t &, cartesian_t &) const;
        void TCNbasis(const cartesian_t &, const cartesian_t &, cartesian_t &,
                      cartesian_t &, cartesian_t &) const;

    private:
        double _a;
        double _e2;
};

isce::core::Ellipsoid& isce::core::Ellipsoid::operator=(const Ellipsoid &rhs) {
    _a = rhs.a();
    _e2 = rhs.e2();
    return *this;
}

double isce::core::Ellipsoid::rEast(double lat) const {
    // Radius of Ellipsoid in East direction (assuming latitude-wise symmetry)
    return _a / std::sqrt(1.0 - (_e2 * std::pow(std::sin(lat), 2)));
}

double isce::core::Ellipsoid::rNorth(double lat) const {
    // Radius of Ellipsoid in North direction (assuming latitude-wise symmetry)
    return (_a * (1.0 - _e2)) / std::pow((1.0 - (_e2 * std::pow(lat, 2))), 1.5);
}

double isce::core::Ellipsoid::rDir(double hdg, double lat) const {
    auto re = rEast(lat);
    auto rn = rNorth(lat);
    return (re * rn) / ((re * std::pow(std::cos(hdg), 2)) 
         + (rn * std::pow(std::sin(hdg), 2)));
}

#endif

// end of file
