// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//
/** \file geometry.h
 * Collection of simple commonly used geometry functions
 *
 * There are no classes defined in this file. Its a collection of functions
 * that are meant to be light weight versions of isce::geometry::Topo and
 * isce::geometry::Geo2rdr.*/

#pragma once

#include "forward.h"

#include <isce/core/forward.h>
#include <isce/product/forward.h>
#include <isce/core/Constants.h>

// Declaration
namespace isce {
//! The isce::geometry namespace
namespace geometry {

/**
 * Radar geometry coordinates to map coordinates transformer
 *
 * This is meant to be the light version of isce::geometry::Topo and not meant
 * to be used for processing large number of targets of interest. Note that
 * doppler and wavelength are meant for completeness and this method can be
 * used with both Native and Zero Doppler geometries. For details of the
 * algorithm, see the \ref overview_geometry "geometry overview".
 *
 * @param[in] aztime azimuth time corresponding to line of interest
 * @param[in] slantRange slant range corresponding to pixel of interest
 * @param[in] doppler doppler model value corresponding to line,pixel
 * @param[in] orbit Orbit object
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] demInterp DEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to aztime and slantRange
 * @param[in] wvl imaging wavelength
 * @param[in] side Left or Right.
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 */
int rdr2geo(double aztime, double slantRange, double doppler,
            const isce::core::Orbit & orbit,
            const isce::core::Ellipsoid & ellipsoid,
            const DEMInterpolator & demInterp,
            isce::core::Vec3 & targetLLH,
            double wvl, isce::core::LookSide side, double threshold,
            int maxIter, int extraIter);

/**
 * Radar geometry coordinates to map coordinates transformer
 *
 * This is the elementary transformation from radar geometry to map
 * geometry. The transformation is applicable for a single slant range
 * and azimuth time (i.e., a single point target). The slant range and
 * Doppler information are encapsulated in the Pixel object, so this
 * function can work for both zero and native Doppler geometries. The
 * azimuth time information is encapsulated in the TCNbasis and
 * StateVector of the platform. For algorithmic details, see
 * \ref overview_geometry "geometry overview".
 *
 * @param[in] pixel Pixel object
 * @param[in] TCNbasis Geocentric TCN basis corresponding to pixel
 * @param[in] pos/vel position and velocity as Vec3 objects
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] demInterp DEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to pixel
 * @param[in] side Left or Right
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 */
int rdr2geo(const isce::core::Pixel & pixel,
            const isce::core::Basis & TCNbasis,
            const isce::core::Vec3& pos,
            const isce::core::Vec3& vel,
            const isce::core::Ellipsoid & ellipsoid,
            const DEMInterpolator & demInterp,
            isce::core::Vec3 & targetLLH,
            isce::core::LookSide side,
            double threshold, int maxIter, int extraIter);

/** "Cone" interface to rdr2geo.
 *
 *  Solve for target position given radar position, range, and cone angle.
 *  The cone is described by a generating axis and the complement of the angle
 *  to that axis (e.g., angle=0 means a plane perpendicular to the axis).  The
 *  vertex of the cone is at the radar position, as is the center of the range
 *  sphere.
 *
 *  Typically `axis` is the velocity vector and `angle` is the squint angle.
 *  However, with this interface you can also set `axis` equal to the long
 *  axis of the antenna, in which case `angle` is an azimuth angle.  In this
 *  manner one can determine where the antenna boresight intersects the ground
 *  at a given range and therefore determine the Doppler from pointing.
 *
 *  @param[in]  radarXYZ  Position of antenna phase center, meters ECEF XYZ.
 *  @param[in]  axis      Cone generating axis (typically velocity), ECEF XYZ.
 *  @param[in]  angle     Complement of cone angle, radians.
 *  @param[in]  range     Range to target, meters.
 *  @param[in]  dem       Digital elevation model, meters above ellipsoid,
 *  @param[out] targetXYZ Target position, ECEF XYZ.
 *  @param[in]  side      Radar look direction
 *  @param[in]  threshold Range convergence threshold, meters.
 *  @param[in]  maxIter   Maximum iterations.
 *  @param[in]  extraIter Additional iterations.
 *
 *  @returns non-zero when iterations successfully converge.
 */
int rdr2geo(const isce::core::Vec3& radarXYZ,
            const isce::core::Vec3& axis, double angle,
            double range, const DEMInterpolator& dem,
            isce::core::Vec3& targetXYZ, isce::core::LookSide side,
            double threshold, int maxIter, int extraIter);

/**
 * Map coordinates to radar geometry coordinates transformer
 *
 * This is the elementary transformation from map geometry to radar geometry.
 * The transformation is applicable for a single lon/lat/h coordinate (i.e., a
 * single point target). For algorithmic details,
 * see \ref overview_geometry "geometry overview".
 *
 * @param[in] inputLLH Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] orbit Orbit object
 * @param[in] doppler   Poly2D Doppler model
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] wavelength Radar wavelength
 * @param[in] startingRange Starting slant range of reference image
 * @param[in] rangePixelSpacing Slant range pixel spacing
 * @param[in] rwidth Width (number of samples) of reference image
 * @param[in] side Left or Right
 * @param[in] threshold azimuth time convergence threshold in seconds
 * @param[in] maxIter Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange step size used for computing derivative of doppler
 */
int geo2rdr(const isce::core::Vec3 & inputLLH,
            const isce::core::Ellipsoid & ellipsoid,
            const isce::core::Orbit & orbit,
            const isce::core::Poly2d & doppler,
            double & aztime, double & slantRange,
            double wavelength, double startingRange,
            double rangePixelSpacing, size_t rwidth, isce::core::LookSide side,
            double threshold, int maxIter, double deltaRange);

/**
 * Map coordinates to radar geometry coordinates transformer
 *
 * This is the elementary transformation from map geometry to radar geometry.
 * The transformation is applicable for a single lon/lat/h coordinate (i.e.,
 * a single point target). For algorithmic details,
 * see \ref overview_geometry "geometry overview".
 *
 * @param[in] inputLLH    Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid   Ellipsoid object
 * @param[in] orbit       Orbit object
 * @param[in] doppler     LUT2d Doppler model
 * @param[out] aztime     azimuth time of inputLLH w.r.t reference epoch of the orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] wavelength  Radar wavelength
 * @param[in] side        Left or Right
 * @param[in] threshold   azimuth time convergence threshold in seconds
 * @param[in] maxIter     Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange  step size used for computing derivative of doppler
 */
int geo2rdr(const isce::core::Vec3 & inputLLH,
            const isce::core::Ellipsoid & ellipsoid,
            const isce::core::Orbit & orbit,
            const isce::core::LUT2d<double> & doppler,
            double & aztime, double & slantRange,
            double wavelength, isce::core::LookSide side, double threshold,
            int maxIter, double deltaRange);

/**
 * Utility function to compute geographic bounds for a radar grid
 *
 * @param[in] orbit     Orbit object
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] doppler   LUT2d doppler object
 * @param[in] lookSide  Left or Right
 * @param[in] radarGrid RadarGridParameters object
 * @param[in] xoff      Column index of radar subwindow
 * @param[in] yoff      Row index of radar subwindow
 * @param[in] xsize     Number of columns of radar subwindow
 * @param[in] ysize     Number of rows of radar subwindiw
 * @param[in] margin    Padding of extracted DEM (radians)
 * @param[out] min_lon  Minimum longitude of geographic region (radians)
 * @param[out] min_lat  Minimum latitude of geographic region (radians)
 * @param[out] max_lon  Maximum longitude of geographic region (radians)
 * @param[out] max_lat  Maximum latitude of geographic region (radians)
 */
void computeDEMBounds(const isce::core::Orbit & orbit,
                      const isce::core::Ellipsoid & ellipsoid,
                      const isce::core::LUT2d<double> & doppler,
                      const isce::product::RadarGridParameters & radarGrid,
                      size_t xoff,
                      size_t yoff,
                      size_t xsize,
                      size_t ysize,
                      double margin,
                      double & min_lon,
                      double & min_lat,
                      double & max_lon,
                      double & max_lat);

template <class T>
double _compute_doppler_aztime_diff(isce::core::Vec3 dr, isce::core::Vec3 satvel,
                                    T & doppler, double wavelength,
                                    double aztime, double slantRange,
                                    double deltaRange);
}
}
