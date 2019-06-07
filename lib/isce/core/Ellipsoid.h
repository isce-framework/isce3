// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_ELLIPSOID_H
#define ISCE_CORE_ELLIPSOID_H
#pragma once

#include "forward.h"

#include <cstdio>
#include <cmath>
#include "Constants.h"

/** Data structure to store Ellipsoid information. 
 *
 * Only the semi-major axis and the eccentricity^2 parameters are stored. All other quantities are derived on the fly*/
class isce::core::Ellipsoid {

    public:
        /** \brief Constructor using semi-major axis and eccentricity^2 
         *
         * @param[in] maj Semi-major of axis in meters
         * @param[in] ecc Square of ellipsoid eccentricity (unitless)*/
        CUDA_HOSTDEV
        Ellipsoid(double maj, double ecc) : _a(maj), _e2(ecc) {}

        /// @cond
        /* Empty constructor - default to Earth WGS-84 ellipsoid */ 
        CUDA_HOSTDEV
        Ellipsoid() : Ellipsoid(EarthSemiMajorAxis, EarthEccentricitySquared) {}
        /// @endcond

        /** Copy constructor*/
        Ellipsoid(const Ellipsoid & ellps) : _a(ellps.a()), _e2(ellps.e2()) {}

        /** Overloaded assignment operator */
        inline Ellipsoid& operator=(const Ellipsoid&);

        /** Return semi-major axis */
        CUDA_HOSTDEV
        double a() const {return _a;}

        /** Return semi-minor axis. Computed from a and e2. */
        CUDA_HOSTDEV
        double b() const {return _a * std::sqrt(1.0 - _e2);}

        /** Return eccentricity^2 */
        CUDA_HOSTDEV
        double e2() const {return _e2;}

        /** \brief Set semi-major axis 
         *
         * @param[in] val Semi-major axis of ellipsoid in meters*/
        void a(double val) {_a = val;}

        /** \brief Set eccentricity^2 
         *
         * @param[in] ecc Eccentricity-squared of ellipsoid*/
        void e2(double val) {_e2 = val;}

        /** Return local radius in EW direction */
        CUDA_HOSTDEV
        inline double rEast(double lat) const;

        /** Return local radius in NS direction */
        CUDA_HOSTDEV
        inline double rNorth(double lat) const;

        /** Return directional local radius */
        CUDA_HOSTDEV
        inline double rDir(double hdg, double lat) const;

        /** Transform WGS84 Lon/Lat/Hgt to ECEF xyz */
        CUDA_HOSTDEV
        void lonLatToXyz(const cartesian_t &llh, cartesian_t &xyz) const;
        CUDA_HOSTDEV Vec3 lonLatToXyz(const Vec3& llh) const {
            Vec3 xyz;
            lonLatToXyz(llh, xyz);
            return xyz;
        }

        /** Transform ECEC xyz to Lon/Lat/Hgt */
        CUDA_HOSTDEV
        void xyzToLonLat(const cartesian_t &xyz, cartesian_t &llh) const;
        CUDA_HOSTDEV Vec3 xyzToLonLat(const Vec3& xyz) const {
            Vec3 llh;
            xyzToLonLat(xyz, llh);
            return llh;
        }

        /** Return normal to the ellipsoid at given lon, lat */
        CUDA_HOSTDEV
        inline void nVector(double lon, double lat, cartesian_t &vec) const;
        CUDA_HOSTDEV
        inline Vec3 nVector(double lon, double lat) const {
            Vec3 result;
            nVector(lon, lat, result);
            return result;
        }

        /** Return ECEF coordinates of point on ellipse */
        CUDA_HOSTDEV
        inline void xyzOnEllipse(double lon, double lat, cartesian_t &xyz) const;

        /** Estimate azimuth angle and look angle for a given LOS vector*/
        void getImagingAnglesAtPlatform(const cartesian_t &pos,const cartesian_t &vel,
                     const cartesian_t &los, double &azi, double &look) const;

    private:
        double _a;
        double _e2;
};

isce::core::Ellipsoid& isce::core::Ellipsoid::operator=(const Ellipsoid &rhs) {
    _a = rhs.a();
    _e2 = rhs.e2();
    return *this;
}

/** @param[in] lat Latitude in radians
 *
 * See <a href="https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical">Prime vertical radius</a>*/
CUDA_HOSTDEV
double isce::core::Ellipsoid::rEast(double lat) const {
    // Radius of Ellipsoid in East direction (assuming latitude-wise symmetry)
    return _a / std::sqrt(1.0 - (_e2 * std::pow(std::sin(lat), 2)));
}


/** @param[in] lat Latitude in radians
 *
 * See <a href="https://en.wikipedia.org/wiki/Earth_radius#Meridional">Meridional radius</a> */
CUDA_HOSTDEV
double isce::core::Ellipsoid::rNorth(double lat) const {
    // Radius of Ellipsoid in North direction (assuming latitude-wise symmetry)
    return (_a * (1.0 - _e2)) / std::pow((1.0 - (_e2 * std::pow(std::sin(lat), 2))), 1.5);
}

/** @param[in] hdg Heading in radians
 *  @param[in] lat Latitude in radians
 *
 *  Heading is measured in clockwise direction from the North direction.
 *  See <a href="https://en.wikipedia.org/wiki/Earth_radius#Directional">Directional Radius</a> */
CUDA_HOSTDEV
double isce::core::Ellipsoid::rDir(double hdg, double lat) const {
    auto re = rEast(lat);
    auto rn = rNorth(lat);
    return (re * rn) / ((re * std::pow(std::cos(hdg), 2)) 
         + (rn * std::pow(std::sin(hdg), 2)));
}


/** @param[in] lon Longitude in radians
 *  @param[in] lat Latitude in radians
 *  @param[out] vec Unit vector of normal pointing outwards in ECEF cartesian coordinates
 *
 *  See <a href="https://en.wikipedia.org/wiki/N-vector">N-vector</a> */
CUDA_HOSTDEV
void isce::core::Ellipsoid::nVector(double lon, double lat, cartesian_t &vec) const
{
    double clat = std::cos(lat);
    vec[0] = clat * std::cos(lon);
    vec[1] = clat * std::sin(lon);
    vec[2] = std::sin(lat);
}


/** @param[in] lon Longitude in radians
 *  @param[in] lat Latitude in radians
 *  @param[out] xyz ECEF coordinates of point on ellipse
 *
 *  See <a href="https://en.wikipedia.org/wiki/Ellipsoid#Parametric_representation">parametric representation of ellipsoid</a>*/
CUDA_HOSTDEV
void isce::core::Ellipsoid::xyzOnEllipse(double lon, double lat, cartesian_t &vec) const
{
    nVector(lon, lat, vec);
    vec[0] *= _a;
    vec[1] *= _a;
    vec[2] *= b();
}

/** @param[in] llh Latitude (ras), Longitude (rad), Height (m).
 *  @param[out] xyz ECEF Cartesian coordinates in meters.*/
CUDA_HOSTDEV inline void isce::core::Ellipsoid::
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
CUDA_HOSTDEV inline void isce::core::Ellipsoid::
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

#endif

// end of file
