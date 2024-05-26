#include "gpuInterpolator.h"

#include <thrust/complex.h>

namespace isce3::cuda::core {

template<class T>
__device__ T gpuNearestNeighborInterpolator<T>::interpolate(
        double x, double y, const T* z, size_t nx, size_t /*ny*/)
{
    size_t x_round = static_cast<size_t>(std::round(x));
    size_t y_round = static_cast<size_t>(std::round(y));

    return z[y_round * nx + x_round];
}

template class gpuNearestNeighborInterpolator<double>;
template class gpuNearestNeighborInterpolator<thrust::complex<double>>;
template class gpuNearestNeighborInterpolator<float>;
template class gpuNearestNeighborInterpolator<thrust::complex<float>>;
template class gpuNearestNeighborInterpolator<unsigned char>;
template class gpuNearestNeighborInterpolator<unsigned short>;
template class gpuNearestNeighborInterpolator<unsigned int>;

} // namespace isce3::cuda::core
