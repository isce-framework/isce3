#include "Device.h"

#include <isce3/cuda/except/Error.h>
#include <isce3/except/Error.h>

namespace isce3 { namespace cuda { namespace core {

static cudaDeviceProp getDeviceProperties(int id)
{
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, id));
    return props;
}

Device::Device(int id) : _id(id)
{
    const int count = getDeviceCount();
    if (id < 0 or id >= count) {
        const std::string errmsg =
                "invalid CUDA device index - " + std::to_string(id);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
}

std::string Device::name() const { return getDeviceProperties(id()).name; }

size_t Device::totalGlobalMem() const
{
    return getDeviceProperties(id()).totalGlobalMem;
}

ComputeCapability Device::computeCapability() const
{
    const auto props = getDeviceProperties(id());
    return {props.major, props.minor};
}

int getDeviceCount()
{
    int count = -1;
    checkCudaErrors(cudaGetDeviceCount(&count));
    return count;
}

Device getDevice()
{
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    return device;
}

void setDevice(Device d) { checkCudaErrors(cudaSetDevice(d.id())); }

}}} // namespace isce3::cuda::core
