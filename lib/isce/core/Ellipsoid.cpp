//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"

/** @param[in] llh Latitude (ras), Longitude (rad), Height (m).
 *  @param[out] xyz ECEF Cartesian coordinates in meters.*/
void isce::core::Ellipsoid::
lonLatToXyz(const cartesian_t & llh, cartesian_t & xyz) const {
    /*
     * Given a lat, lon, and height, produces a geocentric vector.
     */

    // Radius of Earth in East direction
    auto re = rEast(llh[1]);
    // Parametric representation of a circle as a function of longitude
    xyz[0] = (re + llh[2]) * std::cos(llh[1]) * std::cos(llh[0]);
    xyz[1] = (re + llh[2]) * std::cos(llh[1]) * std::sin(llh[0]);
    // Parametric representation with the radius adjusted for eccentricity
    xyz[2] = ((re * (1.0 - _e2)) + llh[2]) * std::sin(llh[1]);
}

/** @param[in] xyz ECEF Cartesian coordinates in meters. 
 *  @param[out] llh Latitude (rad), Longitude(rad), Height (m).
 *
 *  Using the approach laid out in Vermeille, 2002 \cite vermeille2002direct */
void isce::core::Ellipsoid::
xyzToLonLat(const cartesian_t & xyz, cartesian_t & llh) const {
    /*
     * Given a geocentric XYZ, produces a lat, lon, and height above the reference ellipsoid.
     *      VERMEILLE IMPLEMENTATION
     */
    // Pre-compute some values
    const double e4 = _e2 * _e2;
    const double a2 = _a * _a;
    // Lateral distance normalized by the major axis
    double p = (std::pow(xyz[0], 2) + std::pow(xyz[1], 2)) / a2;
    // Polar distance normalized by the minor axis
    double q = ((1. - _e2) * std::pow(xyz[2], 2)) / a2;
    double r = (p + q - e4) / 6.;
    double s = (e4 * p * q) / (4. * std::pow(r, 3));
    double t = std::pow(1. + s + std::sqrt(s * (2. + s)), (1./3.));
    double u = r * (1. + t + (1. / t));
    double rv = std::sqrt(std::pow(u, 2) + (e4 * q));
    double w = (_e2 * (u + rv - q)) / (2. * rv);
    double k = std::sqrt(u + rv + std::pow(w, 2)) - w;
    // Radius adjusted for eccentricity
    double d = (k * std::sqrt(std::pow(xyz[0], 2) + std::pow(xyz[1], 2))) / (k + _e2);
    // Latitude is a function of z and radius
    llh[1] = std::atan2(xyz[2], d);
    // Longitude is a function of x and y
    llh[0] = std::atan2(xyz[1], xyz[0]);
    // Height is a function of location and radius
    llh[2] = ((k + _e2 - 1.) * sqrt(std::pow(d, 2) + std::pow(xyz[2], 2))) / k;
}

/*
void Ellipsoid::xyzToLonLat(cartesian_t &xyz, cartesian_t &llh) {
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
    llh[1] = atan(tant);
    llh[0] = atan2(v[1], v[0]);
    llh[2] = (p / std::cos(llh[1])) - rEast(llh[1]);
}
*/

/** @param[in] pos ECEF coordinates of imaging platform in meters 
 *  @param[in] vel ECEF velocity of imaging platform in meters / sec
 *  @param[in] los Line-of-sight (LOS) vector in ECEF coordinates (meters), pointing from platform to target
 *  @param[out] azi Azimuth angle in radians
 *  @param[out] look Look angle in radians
 *
 *  Azimuth angle is defined as angle of the LOS vector from the North Direction in the anti-clockwise direction.
 *  Look angle is defined as angle of the LOS vector and the downward normal at the imaging platform .
 */
void isce::core::Ellipsoid::
getImagingAnglesAtPlatform(const cartesian_t & pos, const cartesian_t & vel,
        const cartesian_t & los, double & azi, double & look) const {
    /*
     * Computes the look vector given the look angle, azimuth angle, and position vector
     */
    //Estimate lat,lon for platform position
    cartesian_t temp;
    xyzToLonLat(pos, temp);

    //Estimate normal to ellipsoid at platform position
    cartesian_t n;
    nVector(temp[0], temp[1], n);
    LinAlg::scale(n, -1.0);

    //Angle subtended with n-vector pointing downwards
    look = std::acos(LinAlg::dot(n, los) / LinAlg::norm(los));
    LinAlg::cross(n, vel, temp);

    //Get Cross 
    cartesian_t c;
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);

    cartesian_t t;
    LinAlg::unitVec(temp, t);
    azi = std::atan2(LinAlg::dot(c, los), LinAlg::dot(t, los));
}

/**@param[in] pos ECEF coordinates of imaging platform in meters
 * @param[in] vel ECEF velocity of imaging platform in meters/ sec
 * @param[out]  t  Tangent unit vector orthogonal to ellipsoid normal and cross track direction
 * @param[out]  c  Cross track unit vector orthogonal to ellipsoid normal and velocity
 * @param[out]  n  Unit vector along normal to ellipsoid pointing downward */
void isce::core::Ellipsoid::
TCNbasis(const cartesian_t & pos, const cartesian_t & vel, cartesian_t & t,
         cartesian_t & c, cartesian_t & n) const {
    /*
     Get TCN basis vectors. Return three separate basis vectors.
     */
    cartesian_t llh;
    xyzToLonLat(pos, llh);

    n = {-std::cos(llh[1]) * std::cos(llh[0]),
         -std::cos(llh[1]) * std::sin(llh[0]),
         -std::sin(llh[1])};

    cartesian_t temp;
    LinAlg::cross(n, vel, temp);
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);
    LinAlg::unitVec(temp, t);
}

/**@param[in] pos ECEF coordinates of imaging platform in meters
 * @param[in] vel ECEF velocity of imaging platform in meters/sec
 * @param[out] basis t,c,n unit vectors in a basis object
 *
 * See also isce::core::Ellipsoid::TCNbasis(const cartesian_t&, const cartesian_t& vel,
 *                              cartestian_t&, cartesian_t&, cartesian_t&)*/
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
