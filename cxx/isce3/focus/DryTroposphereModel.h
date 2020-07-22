#pragma once

#include <isce3/core/forward.h>

#include <isce3/core/Common.h>
#include <string>

namespace isce3 {
namespace focus {

/**
 * Models used for estimating the propagation delay due to the dry component
 * of the troposphere
 */
enum class DryTroposphereModel {
    /** Don't apply dry troposphere delay term */
    NoDelay = 0,

    /** Estimate dry troposphere delay using TerrSAR-X model \cite breit2010 */
    TSX,
};

/** Convert to string */
std::string toString(DryTroposphereModel);

/**
 * Convert from string
 *
 * \param[in] s Input string in {"nodelay", "tsx"}
 * \returns     DryTroposphereModel enumeration value
 */
DryTroposphereModel parseDryTropoModel(const std::string& s);

/**
 * Estimate dry tropospheric path delay using the TerraSAR-X model
 * \cite breit2010
 *
 * \param[in] p         Antenna phase center position (ECEF m)
 * \param[in] llh       Target Lon/Lat/HAE (deg/deg/m)
 * \param[in] ellipsoid Reference ellipsoid
 * \returns             Propagation delay (s)
 */
CUDA_HOSTDEV
double dryTropoDelayTSX(const isce3::core::Vec3& p, const isce3::core::Vec3& llh,
                        const isce3::core::Ellipsoid& ellipsoid);

} // namespace focus
} // namespace isce3

#include "DryTroposphereModel.icc"
