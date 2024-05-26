#include "gpuFilter.h"
#include "isce3/io/Raster.h"
#include <isce3/cuda/except/Error.h>

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

namespace isce3::cuda::signal {

template<class T>
__global__ void filter_g(thrust::complex<T> *signal, thrust::complex<T> *filter, int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        signal[i] *= filter[i];
    }
}

template<class T>
gpuFilter<T>::~gpuFilter()
{
    if (_filter_set) {
        checkCudaErrors(cudaFree(_d_filter));
    }
}

// do all calculations in place with data stored on device within signal
template<class T>
void gpuFilter<T>::
filter(gpuSignal<T> &signal)
{
    signal.forward();

    auto n_signal_elements = signal.getNumElements();

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_signal_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(signal.getDevicePtr(),
                              _d_filter,
                              n_signal_elements);

    checkCudaErrors(cudaDeviceSynchronize());

    signal.inverse();
}


// pass in device pointer to filter on
template<class T>
void gpuFilter<T>::
filter(thrust::complex<T> *data)
{
    _signal.forwardDevMem(data);

    auto n_signal_elements = _signal.getNumElements();

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_signal_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(data,
                              _d_filter,
                              n_signal_elements);

    checkCudaErrors(cudaDeviceSynchronize());

    _signal.inverseDevMem(data);
}


// pass in host memory to copy to device to be filtered
// interim spectrum is saved as well
template<class T>
void gpuFilter<T>::
filter(std::valarray<std::complex<T>> &signal,
        std::valarray<std::complex<T>> &spectrum)
{
    _signal.dataToDevice(signal);
    _signal.forward();

    // save spectrum
    _signal.dataToHost(spectrum);

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((signal.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(_signal.getDevicePtr(),
                              _d_filter,
                              signal.size());

    checkCudaErrors(cudaDeviceSynchronize());

    _signal.inverse();

    // copy signal to host
    _signal.dataToHost(signal);
}

template<class T>
void gpuFilter<T>::
writeFilter(size_t ncols, size_t nrows)
{
    isce3::io::Raster filterRaster("filter.bin", ncols, nrows, 1, GDT_CFloat32, "ENVI");
}

template<class T>
__global__ void phaseShift_g(thrust::complex<T> *slc,
        T *range,
        double pxlSpace,
        T conj,
        double wavelength,
        T wave_div,
        int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        T phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        thrust::complex<T> complex_phase(cos(phase/wave_div), conj*sin(phase/wave_div));
        slc[i] *= complex_phase;
    }
}

template<>
__global__ void phaseShift_g<float>(thrust::complex<float> *slc,
        float *range,
        double pxlSpace,
        float conj,
        double wavelength,
        float wave_div,
        int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        float phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        thrust::complex<float> complex_phase(cosf(phase/wave_div), conj*sinf(phase/wave_div));
        slc[i] *= complex_phase;
    }
}

template<class T>
__global__ void sumSpectrum_g(thrust::complex<T> *spectrum, T *spectrum_sum, int n_rows, int n_cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_cols) {
        for (int i_row = 0; i_row < n_rows; ++i_row) {
            spectrum_sum[i] += abs(spectrum[i_row*n_cols + i]);
        }
    }
}

// DECLARATIONS
template class gpuFilter<float>;

template __global__ void
sumSpectrum_g<float>(thrust::complex<float> *spectrum, float *spectrum_sum, int n_rows, int n_cols);

} // namespace isce3::cuda::signal