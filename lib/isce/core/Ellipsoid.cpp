//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"

using isce::core::Vec3;

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
