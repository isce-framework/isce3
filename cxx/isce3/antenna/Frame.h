// Antenna frame for Spherical grid
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <isce3/core/Vector.h>

#include "SphGridType.h"

namespace isce3 { namespace antenna {

/**
 * A class for antenna frame and spherical-cartesian
 * coordinate transformation.
 */
class Frame {
public:
    // type aliases
    using Vec3_t = isce3::core::Vec3;
    using Vec2_t = isce3::core::Vector<2>;

    /**
     * Default empty Constructor.
     * The spherical grid type is set to EL_AND_AZ.
     */
    Frame() = default;

    /**
     * Constructor from enum Spherical Grid type
     * @param[in] grid_type : an enum "SphGridType"
     */
    constexpr Frame(const SphGridType& grid_type) : _grid_type(grid_type) {}

    /**
     * Constructor from a string grid type
     * @param[in] grid_str : case-insensitive string representing
     * spherical grid type for spehrical coordinate
     * possible options  : EL_AND_AZ, EL_OVER_AZ, AZ_OVER_EL
     * and THETA_PHI
     * @exception : InvalidArgument
     */
    explicit Frame(const std::string& grid_str)
        : _grid_type(gridTypeFromStr(grid_str))
    {}
    /**
     * Get grid type enum value
     * @return a value from enum 'SphGridType'
     * For grid definitions, see section 6.2 of reference
     * @cite ReeTechDesDoc
     */
    constexpr SphGridType gridType() const { return _grid_type; }

    /**
     * Convert from spherical coordinate with unit radius into cartesian.
     * @param[in] el_theta : scalar, Elevation or Theta angle in (rad)
     * @param[in] az_phi   : scalar, Azimuth or Phi angle in (rad)
     * @param[out] Column-wise Eigen Vector3d or isce3 Vec3 : a unit vector
     * (x,y,z).
     * For equations, see section 6.2 of reference
     * @cite ReeTechDesDoc
     */
    Vec3_t sphToCart(double el_theta, double az_phi) const;

    /**
     * An overloaded function to convert a vector from spherical coordinate
     * with unit radius into cartesian.
     * @param[in] el_theta : a STD vector of Elevation or Theta angles in (rad)
     * @param[in] az_phi   : a STD vector, Azimuth or Phi angle in (rad) , must
     * have the same size as el_theta.
     * @param[out] a STD vector of Column-wise Eigen Vector3d or isce3 Vec3
     * @exception LengthError
     * For equations, see section 6.2 of reference
     * @cite ReeTechDesDoc
     */
    std::vector<Vec3_t> sphToCart(const std::vector<double>& el_theta,
            const std::vector<double>& az_phi) const;

    /**
     * An overloaded function to convert Vector/Scalar values from spherical
     * coordinate with unit radius into cartesian in 3-by-N Eigen  Matrix form.
     * This function is suitable for EL-cut pattern at fixed AZ.
     * @param[in] el_theta : a STD vector of Elevation or Theta
     * angles in (rad).
     * @param[in] az_phi   : a double scalar , Azimuth or Phi angle
     * in (rad).
     * @param[out] a STD vector of Column-wise Eigen Vector3d or isce3 Vec3
     * size of the el_theta.
     */
    std::vector<Vec3_t> sphToCart(
            const std::vector<double>& el_theta, double az_phi) const;

    /**
     * An overloaded function to convert Vector/Scalar values from spherical
     * coordinate with unit radius into cartesian in 3-by-N Eigen  Matrix form.
     * This function is suitable for AZ-cut pattern at fixed EL.
     * @param[in] el_theta : a double scalar , Elevation or Theta angle
     * in (rad).
     * @param[in] az_phi   : a STD vector of Azimuth or Phi angles in (rad).
     * @param[out] a STD vector of Column-wise Eigen Vector3d or isce3 Vec3
     * size of the el_theta.
     */
    std::vector<Vec3_t> sphToCart(
            double el_theta, const std::vector<double>& az_phi) const;

    /**
     * Convert from cartesian into spherical coordinate with unit radius
     * @param[in] vec : Column-wise Eigen Vector3d  or isce3 Vec3 ,
     * a vector (x,y,z)
     * if not a unit vector it will be normalized
     * @param[out] Two-element Eigen double Vector of either (EL, AZ) or
     * (Theta, Phi) in (rad,rad) depending on the grid type.
     */
    Vec2_t cartToSph(Vec3_t vec) const;

    /**
     * An overloaded function to convert a vector from cartesian into spherical
     * coordinate with unit radius
     * @param[in] vec : a STD vector of Column-wise Eigen Vector3d  or isce3
     * Vec3 , a vector (x,y,z). if not a unit vector, it will be normalized.
     * @param[out] a STD vector of Two-element Eigen double Vector or isce3
     * Vector<2> of either (EL, AZ) or (Theta, Phi) in (rad,rad) depending on
     * the grid type.
     */
    std::vector<Vec2_t> cartToSph(std::vector<Vec3_t> vec) const;

private:
    SphGridType _grid_type {SphGridType::EL_AND_AZ};
};

/**
   Operator "==" for equality between two ant frames
   @param[in] lhs : Frame object
   @param[in] rhs : Frame object
   @return bool
*/
bool operator==(const Frame& lhs, const Frame& rhs);

/**
   Operator "!=" for inequality between two ant frames
   @param[in] lhs : Frame object
   @param[in] rhs : Frame object
   @return bool
*/
bool operator!=(const Frame& lhs, const Frame& rhs);

}} // namespace isce3::antenna

#include "Frame.icc"
