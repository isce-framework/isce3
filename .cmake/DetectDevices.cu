#include <cstdio>

int main()
{
    // get number of CUDA devices
    cudaError_t status;
    int count = 0;
    status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return 1;
    }

    // print compute capability for each device
    cudaDeviceProp prop;
    for (int device = 0; device < count; ++device) {
        status = cudaGetDeviceProperties(&prop, device);
        if (status != cudaSuccess) { continue; }
        std::printf("%d.%d ", prop.major, prop.minor);
    }

    return 0;
}
