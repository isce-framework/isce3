/** @file geometryfunc.h
 * A collection of antenna-related geometry functions.
 */
#pragma once

#include <isce3/core/forward.h>

#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <isce3/antenna/Frame.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/geometry/DEMInterpolator.h>

/** @namespace isce3::antenna */
namespace isce3 { namespace antenna {

// Antenna to Radar

/**
 * Estimate Radar products, Slant range and Doppler centroid, from
 * spherical angles in antenna body-fixed domain for a certain spacecraft
 * position, velocity,and attitude at a certain height w.r.t. an ellipsoid.
 * @param[in] el_theta : either elevation or theta angle in radians
 * depending on the "frame" object.
 * @param[in] az_phi : either azimuth or phi angle in radians depending
 * on the "frame" object.
 * @param[in] pos_ecef : antenna/spacecraft position in ECEF (m,m,m)
 * @param[in] vel_ecef : spacecraft velocity in ECEF (m/s,m/s,m/s)
 * @param[in] quat : isce3 quaternion object for transformation from antenna
 * body-fixed to ECEF
 * @param[in] wavelength : Radar wavelength in (m).
 * @param[in] dem_interp (optional): isce3 DEMInterpolator object
 * w.r.t ellipsoid. Default is zero height.
 * @param[in] abs_tol (optional): Abs error/tolerance in height estimation (m)
 * between desired input height and final output height. Default is 0.5.
 * @param[in] max_iter (optional): Max number of iterations in height
 * estimation. Default is 10.
 * @param[in] frame (optional): isce3 Frame object to define antenna spherical
 *  coordinate system. Default is based on "EL_AND_AZ" spherical grid.
 * @param[in] ellips (optional): isce3 Ellipsoid object defining the
 * ellipsoidal planet. Default is WGS84 ellipsoid.
 * @return slantrange (m)
 * @return Doppler (Hz)
 * @return a bool which is true if height tolerance is met, false otherwise.
 * @exception InvalidArgument, RuntimeError
 * @cite ReeTechDesDoc
 */
std::tuple<double, double, bool> ant2rgdop(double el_theta, double az_phi,
        const isce3::core::Vec3& pos_ecef, const isce3::core::Vec3& vel_ecef,
        const isce3::core::Quaternion& quat, double wavelength,
        const isce3::geometry::DEMInterpolator& dem_interp = {},
        double abs_tol = 0.5, int max_iter = 10,
        const isce3::antenna::Frame& frame = {},
        const isce3::core::Ellipsoid& ellips = {});

/**
 * Overloaded function to estimate Radar products, Slant ranges and Doppler
 * centroids, from spherical angles in antenna body-fixed domain for a
 * certain spacecraft position, velocity,and attitude at a certain height
 * w.r.t. an ellipsoid.
 * @param[in] el_theta : a vector of either elevation or theta angle in
 * radians depending on the "frame" object.
 * @param[in] az_phi : either azimuth or phi angle in radians depending
 * on the "frame" object.
 * @param[in] pos_ecef : antenna/spacecraft position in ECEF (m,m,m)
 * @param[in] vel_ecef : spacecraft velocity in ECEF (m/s,m/s,m/s)
 * @param[in] quat : isce3 quaternion object for transformation from antenna
 * body-fixed to ECEF
 * @param[in] wavelength : Radar wavelength in (m).
 * @param[in] dem_interp (optional): isce3 DEMInterpolator object
 * w.r.t ellipsoid. Default is zero height.
 * @param[in] abs_tol (optional): Abs error/tolerance in height estimation (m)
 * between desired input height and final output height. Default is 0.5.
 * @param[in] max_iter (optional): Max number of iterations in height
 * estimation. Default is 10.
 * @param[in] frame (optional): isce3 Frame object to define antenna spherical
 *  coordinate system. Default is based on "EL_AND_AZ" spherical grid.
 * @param[in] ellips (optional): isce3 Ellipsoid object defining the
 * ellipsoidal planet. Default is WGS84 ellipsoid.
 * @return an Eigen::VectorXd of slant ranges (m)
 * @return an Eigen::VectorXd of Doppler values (Hz)
 * @return a bool which is true if height tolerance is met, false otherwise.
 * @exception InvalidArgument, RuntimeError
 * @cite ReeTechDesDoc
 */
std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool> ant2rgdop(
        const Eigen::Ref<const Eigen::VectorXd>& el_theta, double az_phi,
        const isce3::core::Vec3& pos_ecef, const isce3::core::Vec3& vel_ecef,
        const isce3::core::Quaternion& quat, double wavelength,
        const isce3::geometry::DEMInterpolator& dem_interp = {},
        double abs_tol = 0.5, int max_iter = 10,
        const isce3::antenna::Frame& frame = {},
        const isce3::core::Ellipsoid& ellips = {});

// Antenna to Geometry

/**
 * Estimate geodetic geolocation (longitude, latitude, height) from
 * spherical angles in antenna body-fixed domain for a certain spacecraft
 * position and attitude at a certain height w.r.t. an ellipsoid.
 * @param[in] el_theta : either elevation or theta angle in radians
 * depending on the "frame" object.
 * @param[in] az_phi : either azimuth or phi angle in radians depending
 * on the "frame" object.
 * @param[in] pos_ecef : antenna/spacecraft position in ECEF (m,m,m)
 * @param[in] quat : isce3 quaternion object for transformation from antenna
 * body-fixed to ECEF
 * @param[in] dem_interp (optional): isce3 DEMInterpolator object
 * w.r.t ellipsoid. Default is zero height.
 * @param[in] abs_tol (optional): Abs error/tolerance in height estimation (m)
 * between desired input height and final output height. Default is 0.5.
 * @param[in] max_iter (optional): Max number of iterations in height
 * estimation. Default is 10.
 * @param[in] frame (optional): isce3 Frame object to define antenna spherical
 *  coordinate system. Default is based on "EL_AND_AZ" spherical grid.
 * @param[in] ellips (optional): isce3 Ellipsoid object defining the
 * ellipsoidal planet. Default is WGS84 ellipsoid.
 * @return an isce3::core::Vec3 of geodetic LLH(longitude,latitude,height)
 * in (rad,rad,m)
 * @return a bool which is true if height tolerance is met, false otherwise.
 * @exception InvalidArgument, RuntimeError
 * @cite ReeTechDesDoc
 */
std::tuple<isce3::core::Vec3, bool> ant2geo(double el_theta, double az_phi,
        const isce3::core::Vec3& pos_ecef, const isce3::core::Quaternion& quat,
        const isce3::geometry::DEMInterpolator& dem_interp = {},
        double abs_tol = 0.5, int max_iter = 10,
        const isce3::antenna::Frame& frame = {},
        const isce3::core::Ellipsoid& ellips = {});

/**
 * Overloaded function to estimate geodetic geolocation
 * (longitude, latitude, height) from spherical angles in antenna
 * body-fixed domain for a certain spacecraft position and attitude
 * at a certain height w.r.t. an ellipsoid.
 * @param[in] el_theta : a vector of either elevation or theta angle
 * in radians depending on the "frame" object.
 * @param[in] az_phi : either azimuth or phi angle in radians depending
 * on the "frame" object.
 * @param[in] pos_ecef : antenna/spacecraft position in ECEF (m,m,m)
 * @param[in] quat : isce3 quaternion object for transformation from antenna
 * body-fixed to ECEF
 * @param[in] dem_interp (optional): isce3 DEMInterpolator object
 * w.r.t ellipsoid. Default is zero height.
 * @param[in] abs_tol (optional): Abs error/tolerance in height estimation (m)
 * between desired input height and final output height. Default is 0.5.
 * @param[in] max_iter (optional): Max number of iterations in height
 * estimation. Default is 10.
 * @param[in] frame (optional): isce3 Frame object to define antenna spherical
 *  coordinate system. Default is based on "EL_AND_AZ" spherical grid.
 * @param[in] ellips (optional): isce3 Ellipsoid object defining the
 * ellipsoidal planet. Default is WGS84 ellipsoid.
 * @return a vector of isce3::core::Vec3 of geodetic
 * LLH(longitude,latitude,height) in (rad,rad,m)
 * @return a bool which is true if height tolerance is met, false otherwise.
 * @exception InvalidArgument, RuntimeError
 * @cite ReeTechDesDoc
 */
std::tuple<std::vector<isce3::core::Vec3>, bool> ant2geo(
        const Eigen::Ref<const Eigen::VectorXd>& el_theta, double az_phi,
        const isce3::core::Vec3& pos_ecef, const isce3::core::Quaternion& quat,
        const isce3::geometry::DEMInterpolator& dem_interp = {},
        double abs_tol = 0.5, int max_iter = 10,
        const isce3::antenna::Frame& frame = {},
        const isce3::core::Ellipsoid& ellips = {});

/** Compute target position given range and AZ angle by varying EL until height
 *  matches DEM.
 *
 * @param[in] slant_range   Range to target in m
 * @param[in] az            AZ angle in rad
 * @param[in] pos_ecef      ECEF XYZ position of radar in m
 * @param[in] quat          Orientation of the antenna (RCS to ECEF quaternion)
 * @param[in] dem_interp    Digital elevation model in m above ellipsoid
 * @param[in] el_min        Lower bound for EL solution in rad (default=-45 deg)
 * @param[in] el_max        Upper bound for EL solution in rad (default=+45 deg)
 * @param[in] el_tol        Allowable absolute error in EL solution in rad
 *                          Zero for maximum possible precision.  (default=0)
 * @param[in] frame         Coordinate convention for (EL, AZ) to cartesian
 *                          transformation.  (default=EL_AND_AZ)
 *
 * @returns Target position in ECEF in m
 */
isce3::core::Vec3 rangeAzToXyz(double slant_range, double az,
        const isce3::core::Vec3& pos_ecef, const isce3::core::Quaternion& quat,
        const isce3::geometry::DEMInterpolator& dem_interp = {},
        double el_min = -M_PI / 4, double el_max = M_PI / 4,
        double el_tol = 0.0, const isce3::antenna::Frame& frame = {});

}} // namespace isce3::antenna
