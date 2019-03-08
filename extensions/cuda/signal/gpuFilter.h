//
// Author: Liang Yu
// Copyright 2019
//

#ifndef __ISCE_CUDA_SIGNAL_GPUFILTER_H__
#define __ISCE_CUDA_SIGNAL_GPUFILTER_H__

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#define CUDA_GLOBAL
#endif

#include <complex>
#include <valarray>

#include "gpuSignal.h"
#include "isce/cuda/core/gpuComplex.h"
#include <isce/core/LUT1d.h>

using isce::cuda::signal::gpuSignal;
using isce::cuda::core::gpuComplex;

// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            template<class T>
            class gpuFilter;
        }
    }
}

// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            template<class T>
            class gpuAzimuthFilter;
        }
    }
}

// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            template<class T>
            class gpuRangeFilter;
        }
    }
}

using isce::cuda::signal::gpuFilter;
using isce::cuda::signal::gpuAzimuthFilter;
using isce::cuda::signal::gpuRangeFilter;

// Definition of base class
template<class T>
class gpuFilter {
    public:
        gpuFilter() {};
        ~gpuFilter();

        /** Filter a signal in frequency domain*/
        void filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum);

        /** Filter a signal in place on device */
        void filter(gpuSignal<T> &signal);

        /** Filter a signal in place on device */
        void filter(gpuComplex<T> *data);

        /** carry over from parent class. eliminate and use parent? */
        void writeFilter(size_t ncols, size_t nrows);
        
        void cpFilterHostToDevice(std::valarray<std::complex<T>> &host_filter);

    protected:
        
        T *_d_filter;               // device memory pointer
        bool _filter_set = false;
        gpuSignal<T> _signal;
        bool _signal_set = false;
};

// Azimuth filter class derived from base class
template <class T>
class gpuAzimuthFilter : public gpuFilter<T> {
    public:
        //gpuAzimuthFilter() : _d_filter(0x0), _filter_set(false), _signal(), _signal_set(false) {};
        gpuAzimuthFilter();
        ~gpuAzimuthFilter();

        /** constructs forward abd backward FFT plans for filtering a block of data in azimuth direction. */
        void initiateAzimuthFilter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                size_t ncols,
                size_t nrows);

        void constructAzimuthCommonbandFilter(const isce::core::LUT1d<double> & refDoppler,
                const isce::core::LUT1d<double> & secDoppler,
                double bandwidth,
                double prf,
                double beta,
                std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                size_t ncols,
                size_t nrows);
};

// Range filter class derived from base class
template <class T>
class gpuRangeFilter : public gpuFilter<T> {
    public:
        gpuRangeFilter();
        ~gpuRangeFilter();

        // same name wrappers for filter init and construction functions with HostToDevice cp
        /** constructs forward abd backward FFT plans for filtering a block of data in range direction. */
        void initiateRangeFilter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                size_t ncols,
                size_t nrows);

        /** Construct range band-pass filter*/
        void constructRangeBandpassFilter(double rangeSamplingFrequency,
                std::valarray<double> subBandCenterFrequencies,
                std::valarray<double> subBandBandwidths,
                std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                size_t ncols,
                size_t nrows,
                std::string filterType);

        void constructRangeBandpassFilter(double rangeSamplingFrequency,
                std::valarray<double> subBandCenterFrequencies,
                std::valarray<double> subBandBandwidths,
                size_t ncols,
                size_t nrows,
                std::string filterType);

        /** Construct a box car range band-pass filter for multiple bands*/
        void constructRangeBandpassBoxcar(std::valarray<double> subBandCenterFrequencies,
                std::valarray<double> subBandBandwidths,
                double dt,
                int nfft,
                std::valarray<std::complex<T>> &_filter1D);

        void constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                std::valarray<double> subBandBandwidths,
                double dt,
                std::valarray<double>& frequency,
                double beta,
                std::valarray<std::complex<T>>& _filter1D);

        void filterCommonRangeBand(T *d_refSlc, T *d_secSlc, T *range);

        size_t rangeFrequencyShiftMaxIdx(gpuComplex<T> *spectrum,
                int n_rows, 
                int n_cols);

        void getPeakIndex(std::valarray<float> data, size_t &peakIndex);

    private:
        double _wavelength;
        double _rangePixelSpacing;
        double _freqShift;
        double _rangeBandWidth;
        double _rangeSamplingFrequency;
        double _rangeBandwidth;
        T *_d_spectrumSum;
        bool _spectrumSum_set = false;
        std::valarray<T> _spectrumSum;
        std::valarray<std::complex<T>> _filter;
};

template<class T>
__global__ void phaseShift_g(gpuComplex<T> *slc, T *range, double pxlSpace, T conj, double wavelength, T wave_div, int n_elements);

template<class T>
__global__ void filter_g(gpuComplex<T> *signal, gpuComplex<T> *filter, int n_elements);

template<class T>
__global__ void sumSpectrum_g(gpuComplex<T> *spectrum, T *spectrum_sum, int n_rows, int n_cols);

#endif
