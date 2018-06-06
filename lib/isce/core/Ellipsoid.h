// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_ELLIPSOID_H
#define ISCE_CORE_ELLIPSOID_H

#include <cstdio>
#include <cmath>
#include <array>
#include "Basis.h"

// Declaration
namespace isce {
    namespace core {
        class Ellipsoid;
    }
}

// Ellipsoid declaration
class isce::core::Ellipsoid {

    public:
        /** Constructor using semi-major axis and eccentricity^2 */
        Ellipsoid(double maj, double ecc) : _a(maj), _e2(ecc) {}

        /** Empty constructor - not recommended */ 
        Ellipsoid() : Ellipsoid(0.0, 0.0) {}

        /** Copy constructor*/
        Ellipsoid(const Ellipsoid & ellps) : _a(ellps.a()), _e2(ellps.e2()) {}

        /** Overloaded assignment operator */
        inline Ellipsoid& operator=(const Ellipsoid&);

        /** Return semi-major axis */
        double a() const {return _a;}

        /** Return eccentricity^2 */
        double e2() const {return _e2;}

        /** Set semi-major axis */
        void a(double val) {_a = val;}

        /** Set eccentricity^2 */
        void e2(double val) {_e2 = val;}

        /** Return local radius in EW direction */
        inline double rEast(double lat) const;

        /** Return local radius in NS direction */
        inline double rNorth(double lat) const;

        /** Return directional local radius */
        inline double rDir(double lat, double hdg) const;

        /** Transform WGS84 Lat/Lon/Hgt to ECEF xyz */
        void latLonToXyz(const cartesian_t &llh, cartesian_t &xyz) const;

        /** Transform ECEC xyz to Lat/Lon/Hgt */
        void xyzToLatLon(const cartesian_t &xyz, cartesian_t &llh) const;

        /** Estimate look vector for given state vector, azimuth angle and look angle */
        void getAngs(const cartesian_t &pos,const cartesian_t &vel,
                     const cartesian_t &vec, double &az, double &lk) const;

        /** Projection of a vector on TC plane */
        void getTCN_TCvec(const cartesian_t &pos, const cartesian_t &vel,
                          const cartesian_t &vec, cartesian_t &TCvec) const;

        /** Estimate local TCN basis */
        void TCNbasis(const cartesian_t &pos, const cartesian_t &vel, cartesian_t &t,
                      cartesian_t &c, cartesian_t &n) const;

        /**Estimate local TCN basis */
        void TCNbasis(const cartesian_t &pos, const cartesian_t &vel, Basis &tcn) const;

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
    return (_a * (1.0 - _e2)) / std::pow((1.0 - (_e2 * std::pow(std::sin(lat), 2))), 1.5);
}

double isce::core::Ellipsoid::rDir(double hdg, double lat) const {
    auto re = rEast(lat);
    auto rn = rNorth(lat);
    return (re * rn) / ((re * std::pow(std::cos(hdg), 2)) 
         + (rn * std::pow(std::sin(hdg), 2)));
}

#endif

// end of file
