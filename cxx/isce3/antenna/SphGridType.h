// Spherical Grid Type for Antenna
#pragma once

#include <algorithm>
#include <cctype>
#include <string>

#include <isce3/except/Error.h>

namespace isce3 { namespace antenna {

/**
 * An enum for antenna grid type in spherical coordinate system
 */
enum class SphGridType {
    THETA_PHI,  /**< Theta Phi */
    EL_AND_AZ,  /**< Elevation and Azimuth */
    EL_OVER_AZ, /**< Elevation over Azimuth */
    AZ_OVER_EL  /**< Azimuth over Elevation */
};

/**
 * Convert grid type enum value to a string
 * @param[in] grid_type : enum SphGridType
 * @return : string
 */
std::string toStr(SphGridType grid_type);

/**
 * Helper function to get spherical grid type
 * ,Enum 'SphGridType', from a string.
 * @param[in] str : case-insensitive string for spehrical
 * grid type with possible options : EL_AND_AZ, EL_OVER_AZ,
 * AZ_OVER_EL and THETA_PHI.
 * @return SphGridType enum
 * @exception InvalidArgument
 */
SphGridType gridTypeFromStr(std::string str);

}} // namespace isce3::antenna

#include "SphGridType.icc"
