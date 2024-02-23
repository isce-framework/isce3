// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel , Hirad Ghaemi
// Copyright 2017-2018 , 2020-2021
//
/** \file geometry.h
 * Collection of simple commonly used geometry functions
 *
 * There are no classes defined in this file. Its a collection of functions
 * Some are meant to be light weight versions of isce3::geometry::Topo and
 * isce3::geometry::Geo2rdr. Others are useful for pointing analysis and
 * navigation.*/

#pragma once

#include "forward.h"
#include <isce3/core/forward.h>
#include <isce3/product/forward.h>

#include <optional>
#include <tuple>

#include <Eigen/Dense>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/geometry/DEMInterpolator.h>

// Declaration
namespace isce3 {
//! The isce3::geometry namespace
namespace geometry {

/**
 * Radar geometry coordinates to map coordinates transformer
 *
 * This is meant to be the light version of isce3::geometry::Topo and not meant
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
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to aztime and
 * slantRange
 * @param[in] wvl imaging wavelength
 * @param[in] side Left or Right.
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 */
int rdr2geo(double aztime, double slantRange, double doppler,
        const isce3::core::Orbit& orbit,
        const isce3::core::Ellipsoid& ellipsoid,
        const DEMInterpolator& demInterp, isce3::core::Vec3& targetLLH,
        double wvl, isce3::core::LookSide side, double threshold, int maxIter,
        int extraIter);

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
int rdr2geo(const isce3::core::Pixel& pixel, const isce3::core::Basis& TCNbasis,
        const isce3::core::Vec3& pos, const isce3::core::Vec3& vel,
        const isce3::core::Ellipsoid& ellipsoid,
        const DEMInterpolator& demInterp, isce3::core::Vec3& targetLLH,
        isce3::core::LookSide side, double threshold, int maxIter,
        int extraIter);

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
int rdr2geo(const isce3::core::Vec3& radarXYZ, const isce3::core::Vec3& axis,
        double angle, double range, const DEMInterpolator& dem,
        isce3::core::Vec3& targetXYZ, isce3::core::LookSide side,
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
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the
 * orbit
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
int geo2rdr(const isce3::core::Vec3& inputLLH,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::Orbit& orbit, const isce3::core::Poly2d& doppler,
        double& aztime, double& slantRange, double wavelength,
        double startingRange, double rangePixelSpacing, size_t rwidth,
        isce3::core::LookSide side, double threshold, int maxIter,
        double deltaRange);

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
 * @param[out] aztime     azimuth time of inputLLH w.r.t reference epoch of the
 * orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] wavelength  Radar wavelength
 * @param[in] side        Left or Right
 * @param[in] threshold   azimuth time convergence threshold in seconds
 * @param[in] maxIter     Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange  step size used for computing derivative of doppler
 */
int geo2rdr(const isce3::core::Vec3& inputLLH,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& doppler, double& aztime,
        double& slantRange, double wavelength, isce3::core::LookSide side,
        double threshold, int maxIter, double deltaRange);

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
void computeDEMBounds(const isce3::core::Orbit& orbit,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::LUT2d<double>& doppler,
        const isce3::product::RadarGridParameters& radarGrid, size_t xoff,
        size_t yoff, size_t xsize, size_t ysize, double margin, double& min_lon,
        double& min_lat, double& max_lon, double& max_lat);

template<class T>
double _compute_doppler_aztime_diff(isce3::core::Vec3 dr,
        isce3::core::Vec3 satvel, T& doppler, double wavelength, double aztime,
        double slantRange, double deltaRange);

/**
 * Get unit NED(north,east,down) velocity or unit vector from ECEF
 * velocity or unit vector at a certain geodetic location of spacecraft.
 * @param[in] lon : geodetic longitude in radians
 * @param[in] lat : geodetic latitude in radians
 * @param[in] vel : velocity vector or its unit vector in ECEF (x,y,z)
 * @return NED vector
 * <a href="https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates"
 * target="_blank">See Local Tangent Plane Coordinates</a>
 */
isce3::core::Vec3 nedVector(
        double lon, double lat, const isce3::core::Vec3& vel);

/**
 * Get NWU(north,west,up) velocity or unit vector from ECEF velocity
 * or unit vector at a certain geodetic location of spacecraft
 * @param[in] lon : geodetic longitude in radians
 * @param[in] lat : geodetic latitude in radians
 * @param[in] vel : velocity vector or its unit vector in ECEF (x,y,z)
 * @return NWU vector
 * <a href="https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates"
 * target="_blank">See Local Tangent Plane Coordinates</a>
 */
isce3::core::Vec3 nwuVector(
        double lon, double lat, const isce3::core::Vec3& vel);

/**
 * Get unit ENU(east,north,up) velocity or unit vector from ECEF
 * velocity or unit vector at a certain geodetic location of spacecraft.
 * @param[in] lon : geodetic longitude in radians
 * @param[in] lat : geodetic latitude in radians
 * @param[in] vel : velocity vector or its unit vector in ECEF (x,y,z)
 * @return ENU vector
 * <a href="https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates"
 * target="_blank">See Local Tangent Plane Coordinates</a>
 */
isce3::core::Vec3 enuVector(
        double lon, double lat, const isce3::core::Vec3& vel);

/**
 * Get spacecraft heading/track angle from velocity vector at a certain
 * geodetic location of Spacecraft
 * @param[in] lon : geodetic longitude in radians
 * @param[in] lat : geodetic latitude in radians
 * @param[in] vel : velocity vector or its unit vector in ECEF (x,y,z)
 * @return heading/track angle of spacecraft defined wrt North direction
 * in clockwise direction in radians
 */
double heading(double lon, double lat, const isce3::core::Vec3& vel);

/**
 * Get slant range (m) from platform/antenna position in ECEF (x,y,z)
 * to Reference Ellipsoid given unit look vector (poitning) in ECEF
 * @param[in] pos : a non-zero x,y,z positions of antenna/platform
 * in (m,m,m)
 * @param[in] lkvec : looking/pointing unit vector in ECEF towards
 * planet from Antenna/platform
 * @param[in] ellips (optional) : Ellipsoid object. Default is
 * WGS84 reference ellipsoid
 * @return double scalar slant range in (m)
 * @exception InvalidArgument, RuntimeError
 * See section 6.1 of reference
 * @cite ReeTechDesDoc
 */
double slantRangeFromLookVec(const isce3::core::Vec3& pos,
        const isce3::core::Vec3& lkvec,
        const isce3::core::Ellipsoid& ellips = {});

/**
 * Get an approximatre ECEF, LLH position and respective Slant range
 * at a certain height above the reference ellipsoid of planet for a
 * look vector looking from a certain spacecraft position in ECEF
 * towards the planet.
 * @param[out] sr : slant range (m) to the point on the planet at
 * a certain height.
 * @param[out] tg_pos : ECEF Position (x,y,z) of a point target on the
 * planet at certain height
 * @param[out] llh : geodetic Lon/lat/height Position  of a point on
 * the planet at certain height.
 * @param[in] sc_pos : Spacecraft position in ECEF (x,y,z) all in (m)
 * @param[in] lkvec : look unit vector in ECEF, looking from spacecraft
 * towards the planet.
 * @param[in] dem_interp (optional) : DEMInterpolator object wrt to
 * reference ellipsoid. Default is global 0.0 (m) height.
 * @param[in] hgt_err (optional) : Max error in height estimation (m)
 * between desired input height and final output height.
 * @param[in] num_iter (optional) : Max number of iterations in height
 * estimation
 * @param[in] ellips (optional) : Ellipsoid object. Default is
 * WGS84 reference ellipsoid.
 * @param[in] initial_height (optional): initial height wrt ellipsoid
 * used in the iterative process. If not specified or set to {} or
 * std::nullopt, stats of DEM raster is computed if not already.
 * @return a pair of <int,double> scalars for number of iterations and
 * absolute height error, respectively.
 * @exception InvalidArgument, RuntimeError
 * See section 6.1 of reference
 * @cite ReeTechDesDoc
 */
std::pair<int, double> srPosFromLookVecDem(double& sr,
        isce3::core::Vec3& tg_pos, isce3::core::Vec3& llh,
        const isce3::core::Vec3& sc_pos, const isce3::core::Vec3& lkvec,
        const DEMInterpolator& dem_interp = {}, double hgt_err = 0.5,
        int num_iter = 10, const isce3::core::Ellipsoid& ellips = {},
        std::optional<double> initial_height = {});

/**
 * Estimate look angle (off-nadir angle) and ellipsoidal incidence angle at a
 * desired slant range from orbit (spacecraft/antenna statevector) and at
 * a certain relative azimuth time.
 * Note that this is an approximate closed-form solution where an approximate
 * sphere (like SCH coordinate) to Ellipsoid is formed at a certain spacecraft
 * location and along its heading.
 * Finally,the look angle is calculated in a closed-form expression via
 * "Law of Cosines".
 * <a href="https://en.wikipedia.org/wiki/Law_of_cosines"
 * target="_blank">See Law of Cosines</a>
 * The respective ellipsoidal incidence angle is formed in that approximate
 * sphere by using a fixed height w.r.t its ellipsoid for all look angles via
 * "Law of Sines".
 * <a href="https://en.wikipedia.org/wiki/Law_of_sines" target="_blank">See Law
 * of Sines</a>.
 * @param[in] slant_range: true slant range in meters from antenna phase center
 * (or spacecraft position) to the ground.
 * @param[in] orbit: isce3 orbit object.
 * @param[in] az_time (optional): relative azimuth time in seconds w.r.t
 * reference epoch time of orbit object. If not speficied or set to {} or
 * std::nullopt, the mid time of orbit will be used as azimuth time.
 * @param[in] dem_interp (optional) : DEMInterpolator object wrt to
 * reference ellipsoid. Default is global 0.0 (m) height.
 * @param[in] ellips (optional) : Ellipsoid object. Default is WGS84 reference
 * ellipsoid.
 * @return look angle in radians.
 * @return incidence angles in radians.
 * @exception RuntimeError
 * @note that a fixed DEM height, a mean value over all DEM, is used as a
 * relative height above the local sphere defined by along-track radius of
 * curvature of the ellipsoid. No local slope is taken into acccount in
 * estimating incidience angle!
 */
std::tuple<double, double> lookIncAngFromSlantRange(double slant_range,
        const isce3::core::Orbit& orbit, std::optional<double> az_time = {},
        const DEMInterpolator& dem_interp = {},
        const isce3::core::Ellipsoid& ellips = {});

/**
 * Overloaded vectorized version of estimating look angle (off-nadir angle)
 * and ellipsoidal incidence angle at a
 * desired slant range from orbit (spacecraft/antenna statevector) and at
 * a certain relative azimuth time.
 * Note that this is an approximate closed-form solution where an approximate
 * sphere (like SCH coordinate) to Ellipsoid is formed at a certain spacecraft
 * location and along its heading.
 * Finally,the look angle is calculated in a closed-form expression via
 * "Law of Cosines".
 * <a href="https://en.wikipedia.org/wiki/Law_of_cosines"
 * target="_blank">See Law of Cosines</a>
 * The respective ellipsoidal incidence angle is formed in that approximate
 * sphere by using a fixed height w.r.t its ellipsoid for all look angles via
 * "Law of Sines".
 * <a href="https://en.wikipedia.org/wiki/Law_of_sines" target="_blank">See Law
 * of Sines</a>.
 * @param[in] slant_range: true slant range in meters from antenna phase center
 * (or spacecraft position) to the ground.
 * @param[in] orbit: isce3 orbit object.
 * @param[in] az_time (optional): relative azimuth time in seconds w.r.t
 * reference epoch time of orbit object. If not speficied or set to {} or
 * std::nullopt, the mid time of orbit will be used as azimuth time.
 * @param[in] dem_interp (optional) : DEMInterpolator object wrt to
 * reference ellipsoid. Default is global 0.0 (m) height.
 * @param[in] ellips (optional) : Ellipsoid object. Default is WGS84 reference
 * ellipsoid.
 * @return a vector of look angles in radians.
 * @return a vector of incidence angles in radians.
 * @note that a fixed DEM height, a mean value over all DEM, is used as a
 * relative height above the local sphere defined by along-track radius of
 * curvature of the ellipsoid. No local slope is taken into acccount in
 * estimating incidience angle.
 */
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> lookIncAngFromSlantRange(
        const Eigen::Ref<const Eigen::ArrayXd>& slant_range,
        const isce3::core::Orbit& orbit, std::optional<double> az_time = {},
        const DEMInterpolator& dem_interp = {},
        const isce3::core::Ellipsoid& ellips = {});

/**
 * @param[in] dem: DEMInterpolator object.
 * @return mean height in (m).
 */
double compute_mean_dem(const DEMInterpolator& dem);

} // namespace geometry
} // namespace isce3
