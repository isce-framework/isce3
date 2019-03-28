// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include <cstdio>
#include <valarray>
#include "gpuFilter.h"

#include "isce/signal/Filter.h"
#include "isce/cuda/helper_cuda.h"

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

using isce::cuda::signal::gpuRangeFilter;

template<class T>
gpuRangeFilter<T>::gpuRangeFilter()
{
    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    this->_signal= gpuSignal<T>(fft_type);
}

template<class T>
gpuRangeFilter<T>::~gpuRangeFilter()
{
    if (this->_filter_set) {
        cudaFree(this->_d_filter);
    }
    if (_spectrumSum_set) {
        cudaFree(_d_spectrumSum);
    }
}

template<class T>
void gpuRangeFilter<T>::
initiateRangeFilter(std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &spectrum,
        size_t ncols,
        size_t nrows)
{
    // malloc device memory for eventual max frequency search
    if (!_spectrumSum_set) {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_spectrumSum), ncols*sizeof(T)*2));
        _spectrumSum_set = true;
    }

    // set FFT parameters
    this->_signal.rangeFFT(ncols, nrows);
}

template<class T>
void gpuRangeFilter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
        std::valarray<double> subBandCenterFrequencies,
        std::valarray<double> subBandBandwidths,
        std::valarray<std::complex<T>> &signal,
        std::valarray<std::complex<T>> &spectrum,
        size_t ncols,
        size_t nrows,
        std::string filterType)
{
    constructRangeBandpassFilter(rangeSamplingFrequency,
            subBandCenterFrequencies,
            subBandBandwidths,
            ncols,
            nrows,
            filterType);
}


template<class T>
void gpuRangeFilter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
        std::valarray<double> subBandCenterFrequencies,
        std::valarray<double> subBandBandwidths,
        size_t ncols,
        size_t nrows,
        std::string filterType)
{
    size_t nfft = 0;
    this->_signal.nextPowerOfTwo(ncols, nfft);

    this->_filter.resize(nfft*nrows);
    std::valarray<std::complex<T>> _filter1D(nfft); //
    _filter1D = std::complex<T>(0.0,0.0);

    std::valarray<double> frequency(nfft);
    isce::signal::Filter<float> tempFilter;
    double dt = 1.0/rangeSamplingFrequency;
    isce::signal::Filter<float>::fftfreq(nfft, dt, frequency);

    if (filterType=="boxcar"){
        constructRangeBandpassBoxcar(
                            subBandCenterFrequencies,
                            subBandBandwidths,
                            dt,
                            nfft,
                            _filter1D);

    } else if (filterType=="cosine"){
        double beta = 0.25;
        constructRangeBandpassCosine(subBandCenterFrequencies,
                            subBandBandwidths,
                            dt,
                            frequency,
                            beta,
                            _filter1D);

    } else {
        std::cout << filterType << " filter has not been implemented" << std::endl;
    }

    //construct a block of the filter
    for (size_t line = 0; line < nrows; line++ ){
        for (size_t col = 0; col < nfft; col++ ){
            this->_filter[line*nfft+col] = _filter1D[col];
        }
    }

    this->cpFilterHostToDevice(this->_filter);

    this->_signal.rangeFFT(ncols, nrows);
}

template <class T>
void gpuRangeFilter<T>::
constructRangeBandpassBoxcar(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             int nfft,
                             std::valarray<std::complex<T>>& _filter1D)
{
    // construct a boxcar bandpass filter in frequency domian
    // which may have several bands defined by centerferquencies and
    // subBandBandwidths
    for (size_t i = 0; i<subBandCenterFrequencies.size(); ++i){
        std::cout << "i: " << i << std::endl;
        //frequency of the lower bound of this band
        double fL = subBandCenterFrequencies[i] - subBandBandwidths[i]/2;

        //frequency of the higher bound of this band
        double fH = subBandCenterFrequencies[i] + subBandBandwidths[i]/2;

        //index of frequencies for fL and fH
        int indL;
        isce::signal::Filter<T>::indexOfFrequency(dt, nfft, fL, indL);
        int indH;
        isce::signal::Filter<T>::indexOfFrequency(dt, nfft, fH, indH);
        std::cout << "fL: "<< fL << " , fH: " << fH << " indL: " << indL << " , indH: " << indH << std::endl;
        if (fL<0 && fH>=0){
            for (size_t ind = indL; ind < nfft; ++ind){
                _filter1D[ind] = std::complex<T>(1.0, 0.0);
            }
            for (size_t ind = 0; ind < indH; ++ind){
                _filter1D[ind] = std::complex<T>(1.0, 0.0);
            }

        }else{
            for (size_t ind = indL; ind < indH; ++ind){
                _filter1D[ind] = std::complex<T>(1.0, 0.0);
            }
        }
    }
}

template <class T>
void gpuRangeFilter<T>::
constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D)
{
    const double norm = 1.0;

    for (size_t i = 0; i<subBandCenterFrequencies.size(); ++i){
        double fmid = subBandCenterFrequencies[i];
        double bandwidth = subBandBandwidths[i];
        const double df = 0.5 * bandwidth * beta;
        for (size_t i = 0; i < frequency.size(); ++i) {

            // Get the absolute value of shifted frequency
            const double freq = std::abs(frequency[i] - fmid);

            // Passband
            if (freq <= (0.5 * bandwidth - df)) {
                _filter1D[i] = std::complex<T>(norm, 0.0);

            // Transition region
            } else if (freq > (0.5 * bandwidth - df) && freq <= (0.5 * bandwidth + df)) {
                _filter1D[i] = std::complex<T>(norm * 0.5 *
                                    (1.0 + std::cos(M_PI / (bandwidth*beta) *
                                    (freq - 0.5 * (1.0 - beta) * bandwidth))), 0.0);

            }
        }

    }
}

template<class T>
void gpuRangeFilter<T>::
filterCommonRangeBand(T *d_refSlc, T *d_secSlc, T *range)
{
    auto n_elements = this->_signal.getNumElements();

    // determine block layout; set these in constructor since they're based on n_elements?
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // apply full phase correction to both signals
    phaseShift_g<<<grid, block>>>(reinterpret_cast<thrust::complex<T> *>(&d_refSlc),
            range,
            _rangePixelSpacing,
            T(1.),
            _wavelength,
            T(1.),
            n_elements);
    phaseShift_g<<<grid, block>>>(reinterpret_cast<thrust::complex<T> *>(&d_secSlc),
            range,
            _rangePixelSpacing,
            T(-1.),
            _wavelength,
            T(1.),
            n_elements);

    auto ncols = this->_signal.getColumns();
    auto nrows = this->_signal.getRows();
    std::valarray<double> rangeFrequencies(ncols);
    isce::signal::Filter<float>::fftfreq(ncols, 1.0/_rangeSamplingFrequency, rangeFrequencies);

    // calculate frequency shift
    size_t refIdx = rangeFrequencyShiftMaxIdx(reinterpret_cast<thrust::complex<T> *>(&d_refSlc), nrows, ncols);
    size_t secIdx = rangeFrequencyShiftMaxIdx(reinterpret_cast<thrust::complex<T> *>(&d_secSlc), nrows, ncols);
    double frequencyShift = rangeFrequencies[refIdx] - rangeFrequencies[secIdx];

    std::valarray<double> filterCenterFrequency{0.0};
    std::valarray<double> filterBandwidth{_rangeBandwidth - frequencyShift};
    std::string filterType = "cosine";

    // TODO do this on GPU?
    constructRangeBandpassFilter(_rangeSamplingFrequency,
            filterCenterFrequency,
            filterBandwidth,
            ncols,
            nrows,
            filterType);

    //
    this->filter(reinterpret_cast<thrust::complex<T> *>(&d_refSlc));
    this->filter(reinterpret_cast<thrust::complex<T> *>(&d_secSlc));

    // apply half phase correction
    phaseShift_g<<<grid, block>>>(reinterpret_cast<thrust::complex<T> *>(&d_refSlc),
            range,
            _rangePixelSpacing,
            T(-1.),
            _wavelength,
            T(2.),
            n_elements);
    phaseShift_g<<<grid, block>>>(reinterpret_cast<thrust::complex<T> *>(&d_secSlc),
            range,
            _rangePixelSpacing,
            T(1.),
            _wavelength,
            T(2.),
            n_elements);
}

template<class T>
size_t gpuRangeFilter<T>::
rangeFrequencyShiftMaxIdx(thrust::complex<T> *spectrum,
        int n_rows,
        int n_cols)
{
    int n_elements = n_rows * n_cols;

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // sum spectrum along columns
    sumSpectrum_g<<<grid, block>>>(spectrum, _d_spectrumSum, n_rows, n_cols);

    // copy to signal sums and find index of max value
    checkCudaErrors(cudaMemcpy(&_spectrumSum[0], _d_spectrumSum, n_elements*sizeof(T), cudaMemcpyDeviceToHost));
    size_t idx = 0;
    getPeakIndex(_spectrumSum, idx);
    return idx;
}


template<class T>
void gpuRangeFilter<T>::
getPeakIndex(std::valarray<float> data, size_t &peakIndex)
{
    size_t dataLength = data.size();
    peakIndex = 0;
    double peak = data[peakIndex];
    for (size_t i = 1; i< dataLength;  ++i){
        if (std::abs(data[i]) > peak){
            peak = data[i];
            peakIndex = i;
        }
    }
}

// DECLARATIONS
template class gpuRangeFilter<float>;
