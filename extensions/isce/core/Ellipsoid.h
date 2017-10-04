//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_ELLIPSOID_H__
#define __ISCE_CORE_ELLIPSOID_H__

#include <cmath>
#include <vector>
#include "Constants.h"

namespace isce { namespace core {
    struct Ellipsoid {
        // Major semi-axis
        double a;
        // Eccentricity squared
        double e2;

        Ellipsoid(double maj, double ecc) : a(maj), e2(ecc) {}
        Ellipsoid() : Ellipsoid(0.,0.) {}
        Ellipsoid(const Ellipsoid &e) : a(e.a), e2(e.e2) {}
        inline Ellipsoid& operator=(const Ellipsoid&);

        inline double rEast(double) const;
        inline double rNorth(double) const;
        inline double rDir(double,double) const;
        void latLonToXyz(const std::vector<double>&,std::vector<double>&) const;
        void xyzToLatLon(const std::vector<double>&,std::vector<double>&) const;
        void getAngs(const std::vector<double>&,const std::vector<double>&,
                     const std::vector<double>&,double&,double&) const;
        void getTCN_TCvec(const std::vector<double>&,const std::vector<double>&,
                          const std::vector<double>&,std::vector<double>&) const;
        void TCNbasis(const std::vector<double>&,const std::vector<double>&,std::vector<double>&,
                      std::vector<double>&,std::vector<double>&) const;
    };

    inline Ellipsoid& Ellipsoid::operator=(const Ellipsoid &rhs) {
        a = rhs.a;
        e2 = rhs.e2;
        return *this;
    }

    inline double Ellipsoid::rEast(double lat) const {
        // Radius of Ellipsoid in East direction (assuming latitude-wise symmetry)
        return a / sqrt(1. - (e2 * pow(sin(lat), 2)));
    }

    inline double Ellipsoid::rNorth(double lat) const {
        // Radius of Ellipsoid in North direction (assuming latitude-wise symmetry)
        return (a * (1. - e2)) / pow((1. - (e2 * pow(lat, 2))), 1.5);
    }

    inline double Ellipsoid::rDir(double hdg, double lat) const {
        auto re = rEast(lat);
        auto rn = rNorth(lat);
        return (re * rn) / ((re * pow(cos(hdg), 2)) + (rn * pow(sin(hdg), 2)));
    }
}}

#endif
