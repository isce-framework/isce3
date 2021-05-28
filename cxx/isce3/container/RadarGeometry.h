#pragma once

#include <isce3/core/LUT2d.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3 {
namespace container {

/**
 * Scene geometry in radar coordinate system
 *
 * Encapsulates radar grid, orbit, and Doppler
 */
class RadarGeometry {
private:
    using DateTime = isce3::core::DateTime;
    using LookSide = isce3::core::LookSide;
    using Orbit = isce3::core::Orbit;
    using RadarGridParameters = isce3::product::RadarGridParameters;

    template<typename T> using Linspace = isce3::core::Linspace<T>;
    template<typename T> using LUT2d = isce3::core::LUT2d<T>;

public:
    RadarGeometry(const RadarGridParameters& radar_grid, const Orbit& orbit,
                  const LUT2d<double>& doppler);

    /** Get radar grid */
    const RadarGridParameters& radarGrid() const { return _radar_grid; }

    /** Get platform orbit */
    const Orbit& orbit() const { return _orbit; }

    /** Get Doppler */
    const LUT2d<double>& doppler() const { return _doppler; }

    /** Get reference epoch */
    const DateTime& referenceEpoch() const { return orbit().referenceEpoch(); }

    /** Get radar grid length (number of azimuth lines) */
    size_t gridLength() const { return radarGrid().length(); }

    /** Get radar grid width (number of range samples) */
    size_t gridWidth() const { return radarGrid().width(); }

    /** Get radar grid azimuth time samples relative to reference epoch (s) */
    Linspace<double> sensingTime() const;

    /** Get radar grid slant range samples (m) */
    Linspace<double> slantRange() const;

    /** Get radar look side */
    LookSide lookSide() const { return radarGrid().lookSide(); }

    /** Get radar wavelength */
    double wavelength() const { return radarGrid().wavelength(); }

private:
    RadarGridParameters _radar_grid;
    Orbit _orbit;
    LUT2d<double> _doppler;
};

} // namespace container
} // namespace isce3

#include "RadarGeometry.icc"
