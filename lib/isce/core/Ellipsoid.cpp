//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"

void isce::core::Ellipsoid::
latLonToXyz(const cartesian_t & llh, cartesian_t & xyz) const {
    /*
     * Given a lat, lon, and height, produces a geocentric vector.
     */

    // Radius of Earth in East direction
    auto re = rEast(llh[0]);
    // Parametric representation of a circle as a function of longitude
    xyz[0] = (re + llh[2]) * std::cos(llh[0]) * std::cos(llh[1]);
    xyz[1] = (re + llh[2]) * std::cos(llh[0]) * std::sin(llh[1]);
    // Parametric representation with the radius adjusted for eccentricity
    xyz[2] = ((re * (1.0 - _e2)) + llh[2]) * std::sin(llh[0]);
}

void isce::core::Ellipsoid::
xyzToLatLon(const cartesian_t & xyz, cartesian_t & llh) const {
    /*
     * Given a geocentric XYZ, produces a lat, lon, and height above the reference ellipsoid.
     *      VERMEILLE IMPLEMENTATION
     */
    // Lateral distance normalized by the major axis
    double p = (std::pow(xyz[0], 2) + std::pow(xyz[1], 2)) / std::pow(_a, 2);
    // Polar distance normalized by the minor axis
    double q = ((1. - _e2) * std::pow(xyz[2], 2)) / std::pow(_a, 2);
    double r = (p + q - std::pow(_e2, 2)) / 6.;
    double s = (std::pow(_e2, 2) * p * q) / (4. * std::pow(r, 3));
    double t = std::pow(1. + s + sqrt(s * (2. + s)), (1./3.));
    double u = r * (1. + t + (1. / t));
    double rv = sqrt(std::pow(u, 2) + (std::pow(_e2, 2) * q));
    double w = (_e2 * (u + rv - q)) / (2. * rv);
    double k = sqrt(u + rv + std::pow(w, 2)) - w;
    // Radius adjusted for eccentricity
    double d = (k * sqrt(std::pow(xyz[0], 2) + std::pow(xyz[1], 2))) / (k + _e2);
    // Latitude is a function of z and radius
    llh[0] = atan2(xyz[2], d);
    // Longitude is a function of x and y
    llh[1] = atan2(xyz[1], xyz[0]);
    // Height is a function of location and radius
    llh[2] = ((k + _e2 - 1.) * sqrt(std::pow(d, 2) + std::pow(xyz[2], 2))) / k;
}

/*
void Ellipsoid::xyzToLatLon(cartesian_t &xyz, cartesian_t &llh) {
    //
    // Given a geocentric XYZ, produces a lat, lon, and height above the reference ellipsoid.
    //      SCOTT HENSLEY IMPLEMENTATION
    //

    // Error checking to make sure inputs have expected characteristics
    checkVecLen(llh,3);
    checkVecLen(xyz,3);

    double b = a * sqrt(1. - _e2);
    double p = sqrt(std::pow(v[0], 2) + std::pow(v[1], 2));
    double tant = (v[2] / p) * sqrt(1. / (1. - _e2));
    double theta = atan(tant);
    tant = (v[2] + (((1. / (1. - _e2)) - 1.) * b * std::pow(std::sin(theta), 3))) /
           (p - (_e2 * a * std::pow(std::cos(theta), 3)));
    llh[0] = atan(tant);
    llh[1] = atan2(v[1], v[0]);
    llh[2] = (p / std::cos(llh[0])) - rEast(llh[0]);
}
*/

void isce::core::Ellipsoid::
getAngs(const cartesian_t & pos, const cartesian_t & vel,
        const cartesian_t & vec, double & az, double & lk) const {
    /*
     * Computes the look vector given the look angle, azimuth angle, and position vector
     */
    cartesian_t temp;
    xyzToLatLon(pos, temp);

    cartesian_t n = {-std::cos(temp[0]) * std::cos(temp[1]),
                        -std::cos(temp[0]) * std::sin(temp[1]),
                        -std::sin(temp[0])};
    lk = std::acos(LinAlg::dot(n, vec) / LinAlg::norm(vec));
    LinAlg::cross(n, vel, temp);

    cartesian_t c;
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);

    cartesian_t t;
    LinAlg::unitVec(temp, t);
    az = std::atan2(LinAlg::dot(c, vec), LinAlg::dot(t, vec));
}

void isce::core::Ellipsoid::
getTCN_TCvec(const cartesian_t & pos, const cartesian_t & vel,
             const cartesian_t & vec, cartesian_t & TCVec) const {
    /*
     * Computes the projection of an xyz vector on the TC plane in xyz
     */
    cartesian_t temp;
    xyzToLatLon(pos, temp);

    cartesian_t n = {-std::cos(temp[0]) * std::cos(temp[1]),
                        -std::cos(temp[0]) * std::sin(temp[1]),
                        -std::sin(temp[0])};
    LinAlg::cross(n, vel, temp);

    cartesian_t c;
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);

    cartesian_t t;
    LinAlg::unitVec(temp, t);
    LinAlg::linComb(LinAlg::dot(t, vec), t, LinAlg::dot(c, vec), c, TCVec);
}

void isce::core::Ellipsoid::
TCNbasis(const cartesian_t & pos, const cartesian_t & vel, cartesian_t & t,
         cartesian_t & c, cartesian_t & n) const {
    /*
     Get TCN basis vectors. Return three separate basis vectors.
     */
    cartesian_t llh;
    xyzToLatLon(pos, llh);

    n = {-std::cos(llh[0]) * std::cos(llh[1]),
         -std::cos(llh[0]) * std::sin(llh[1]),
         -std::sin(llh[0])};

    cartesian_t temp;
    LinAlg::cross(n, vel, temp);
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);
    LinAlg::unitVec(temp, t);
}

void isce::core::Ellipsoid::
TCNbasis(const cartesian_t & pos, const cartesian_t & vel, Basis & basis) const {
    /*
     Get TCN basis vectors. Return vector in Basis object.
     */
    cartesian_t that, chat, nhat;
    TCNbasis(pos, vel, that, chat, nhat);
    basis.x0(that);
    basis.x1(chat);
    basis.x2(nhat);
}

// end of file
