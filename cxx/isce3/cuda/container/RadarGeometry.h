#pragma once

#include <isce3/container/RadarGeometry.h>
#include <isce3/cuda/core/Orbit.h>
#include <isce3/cuda/core/gpuLUT2d.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce { namespace cuda { namespace container {

/**
 * CUDA counterpart of isce::container::RadarGeometry
 */
class RadarGeometry {
    using DateTime = isce::core::DateTime;
    using LookSide = isce::core::LookSide;
    using RadarGridParameters = isce::product::RadarGridParameters;

    using HostOrbit = isce::core::Orbit;
    using DeviceOrbit = isce::cuda::core::Orbit;

    using HostRadarGeometry = isce::container::RadarGeometry;

    template<typename T>
    using Linspace = isce::core::Linspace<T>;

    template<typename T>
    using HostLUT2d = isce::core::LUT2d<T>;

    template<typename T>
    using DeviceLUT2d = isce::cuda::core::gpuLUT2d<T>;

public:
    /** Construct a new RadarGeometry object. */
    RadarGeometry(const RadarGridParameters& radar_grid,
                  const DeviceOrbit& orbit, const DeviceLUT2d<double>& doppler);

    /** Construct a new RadarGeometry object. */
    RadarGeometry(const RadarGridParameters& radar_grid, const HostOrbit& orbit,
                  const HostLUT2d<double>& doppler)
        : RadarGeometry(radar_grid, DeviceOrbit(orbit),
                        DeviceLUT2d<double>(doppler))
    {}

    /** Copy a host RadarGeometry object to the device. */
    RadarGeometry(const HostRadarGeometry& other)
        : RadarGeometry(other.radarGrid(), other.orbit(), other.doppler())
    {}

    /** Get radar grid */
    const RadarGridParameters& radarGrid() const { return _radar_grid; }

    /** Get orbit */
    const DeviceOrbit& orbit() const { return _orbit; }

    /** Get Doppler */
    const DeviceLUT2d<double>& doppler() const { return _doppler; }

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

private:
    RadarGridParameters _radar_grid;
    DeviceOrbit _orbit;
    DeviceLUT2d<double> _doppler;
};

}}} // namespace isce::cuda::container

#include "RadarGeometry.icc"
