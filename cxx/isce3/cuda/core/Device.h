#pragma once

#include <string>

#include "ComputeCapability.h"

namespace isce3 { namespace cuda { namespace core {

/** A CUDA-enabled device */
class Device {
public:
    /**
     * Construct a new Device object.
     *
     * Does not change the currently active CUDA device.
     *
     * \throws isce::except::InvalidArgument
     * if the specified device is a not a valid CUDA device
     *
     * \param[in] id Device index (0-based)
     */
    Device(int id);

    /** Return the (0-based) device index. */
    int id() const noexcept { return _id; }

    /** Return a string identifying the device. */
    std::string name() const;

    /** Get the total global memory capacity in bytes. */
    size_t totalGlobalMem() const;

    /** Get the compute capability. */
    ComputeCapability computeCapability() const;

    friend bool operator==(Device lhs, Device rhs) noexcept
    {
        return lhs.id() == rhs.id();
    }

    friend bool operator!=(Device lhs, Device rhs) noexcept
    {
        return not(lhs == rhs);
    }

private:
    int _id;
};

/** Return the number of available CUDA devices. */
int getDeviceCount();

/** Get the current CUDA device for the active host thread. */
Device getDevice();

/** Set the CUDA device for the active host thread. */
void setDevice(Device d);

}}} // namespace isce3::cuda::core
