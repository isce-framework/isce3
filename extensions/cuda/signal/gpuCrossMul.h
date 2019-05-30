// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018-
//

#ifndef ISCE_CUDA_SIGNAL_CROSSMUL_H
#define ISCE_CUDA_SIGNAL_CROSSMUL_H

#include <isce/io/Raster.h>
#include <isce/core/LUT1d.h>
#include <isce/core/Common.h>
#include <isce/cuda/core/gpuLUT1d.h>
#include "gpuSignal.h"
#include "gpuFilter.h"
#include "gpuLooks.h"
#include <thrust/complex.h>


// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            class gpuCrossmul;
        }
    }
}

class isce::cuda::signal::gpuCrossmul {

    public:
        gpuCrossmul() {};
        ~gpuCrossmul() {};

        void crossmul(isce::io::Raster& referenceSLC,
                isce::io::Raster& secondarySLC,
                isce::io::Raster& interferogram,
                isce::io::Raster& coherence);

        void crossmul(isce::io::Raster& referenceSLC,
                isce::io::Raster& secondarySLC,
                isce::io::Raster& rngOffsetRaster,
                isce::io::Raster& interferogram,
                isce::io::Raster& coherenceRaster);

       /** Set doppler LUTs for reference and secondary SLCs*/
        void doppler(isce::core::LUT1d<double>,
                isce::core::LUT1d<double>);

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double p_r_f) {_prf = p_r_f;};

        /** Set range sampling frequency  */
        inline void rangeSamplingFrequency(double rngSampV) {_rangeSamplingFrequency = rngSampV;};

        /** Set the range bandwidth */
        inline void rangeBandwidth(double rngBW) {_rangeBandwidth = rngBW;};

        /** Range pixel spacing */
        inline void rangePixelSpacing(double rngPxl) {_rangePixelSpacing = rngPxl;};

        /** Set Wavelength*/
        inline void wavelength(double v) {_wavelength = v;};

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double azBW) {_commonAzimuthBandwidth = azBW;};

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double b) {_beta = b;};

        /** Set number of range looks */
        void rangeLooks(int rngLks);

        /** Set number of azimuth looks */
        void azimuthLooks(int azLks);

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthBandFiltering(bool doAz) {_doCommonAzimuthBandFilter = doAz;};

        /** Set common range band filtering flag */
        inline void doCommonRangeBandFiltering(bool doRng) {_doCommonRangeBandFilter = doRng;};

    private:
        //Doppler LUT for the refernce SLC
        isce::core::LUT1d<double> _refDoppler;

        //Doppler LUT for the secondary SLC
        isce::core::LUT1d<double> _secDoppler;

        //pulse repetition frequency
        double _prf;

        // range samping frequency
        double _rangeSamplingFrequency;

        // range signal bandwidth
        double _rangeBandwidth;

        // range pixel spacing
        double _rangePixelSpacing;

        // radar wavelength
        double _wavelength;

        //azimuth common bandwidth
        double _commonAzimuthBandwidth;

        // beta parameter for constructing common azimuth band filter
        double _beta;

        // number of range looks
        int _rangeLooks = 1;

        // number of azimuth looks
        int _azimuthLooks = 1;

        bool _doMultiLook = false;

        // Flag for common azimuth band filtering
        bool _doCommonAzimuthBandFilter = false;

        // Flag for common range band filtering
        bool _doCommonRangeBandFilter = false;

        // number of lines per block
        size_t blockRows = 8192;

        // upsampling factor
        size_t oversample = 1;
};

void lookdownShiftImpact(size_t oversample,
        size_t nfft,
        size_t blockRows,
        std::valarray<std::complex<float>> &shiftImpact);

template <class T>
CUDA_GLOBAL void interferogram_g(thrust::complex<T> *ifgram,
        thrust::complex<T> *refSlcUp,
        thrust::complex<T> *secSlcUp,
        int n_rows,
        int n_cols,
        int n_fft,
        int oversample_i,
        T oversample_f);

template <class T>
CUDA_GLOBAL void calculate_coherence_g(T *ref_amp,
        T *sec_amp,
        thrust::complex<T> *ifgram_mlook,
        int n_elements);
#endif
