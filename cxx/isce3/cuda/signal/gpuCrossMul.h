// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018-

#pragma once

#include "forward.h"
#include <isce3/io/forward.h> // Raster

#include <isce3/core/Common.h>
#include <isce3/core/LUT1d.h>
#include <thrust/complex.h>

class isce3::cuda::signal::gpuCrossmul {

    public:
        gpuCrossmul() {};
        ~gpuCrossmul() {};

        void crossmul(isce3::io::Raster& referenceSLC,
                isce3::io::Raster& secondarySLC,
                isce3::io::Raster& interferogram,
                isce3::io::Raster& coherence);

        void crossmul(isce3::io::Raster& referenceSLC,
                isce3::io::Raster& secondarySLC,
                isce3::io::Raster& rngOffsetRaster,
                isce3::io::Raster& interferogram,
                isce3::io::Raster& coherenceRaster);

       /** Set doppler LUTs for reference and secondary SLCs*/
        void doppler(isce3::core::LUT1d<double>,
                isce3::core::LUT1d<double>);

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double p_r_f) {_prf = p_r_f;};

        /** Get pulse repetition frequency (PRF) */
        inline double prf() const {return _prf;};

        /** Set range sampling frequency  */
        inline void rangeSamplingFrequency(double rngSampV) {_rangeSamplingFrequency = rngSampV;};

        /** Get range sampling frequency  */
        inline double rangeSamplingFrequency() const {return _rangeSamplingFrequency;};

        /** Set the range bandwidth */
        inline void rangeBandwidth(double rngBW) {_rangeBandwidth = rngBW;};

        /** Get the range bandwidth */
        inline double rangeBandwidth() const {return _rangeBandwidth;};

        /** Set range pixel spacing */
        inline void rangePixelSpacing(double rngPxl) {_rangePixelSpacing = rngPxl;};

        /** Get range pixel spacing */
        inline double rangePixelSpacing() const {return _rangePixelSpacing;};

        /** Set Wavelength*/
        inline void wavelength(double v) {_wavelength = v;};

        /** Get Wavelength*/
        inline double wavelength() const {return _wavelength;};

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double azBW) {_commonAzimuthBandwidth = azBW;};

        /** Get azimuth common bandwidth */
        inline double commonAzimuthBandwidth() const {return _commonAzimuthBandwidth;};

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double b) {_beta = b;};

        /** Get beta parameter for the azimuth common band filter */
        inline double beta() const {return _beta;};

        /** Set number of range looks */
        void rangeLooks(int rngLks);

        /** Get number of range looks */
        inline int rangeLooks() const {return _rangeLooks;};

        /** Set number of azimuth looks */
        void azimuthLooks(int azLks);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const {return _azimuthLooks;};

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthBandFilter(bool doAz) {_doCommonAzimuthBandFilter = doAz;};

        /** Get common azimuth band filtering flag */
        inline bool doCommonAzimuthBandFilter() const {return _doCommonAzimuthBandFilter;};

        /** Set common range band filtering flag */
        inline void doCommonRangeBandFilter(bool doRng) {_doCommonRangeBandFilter = doRng;};

        /** Get common range band filtering flag */
        inline bool doCommonRangeBandFilter() const {return _doCommonRangeBandFilter;};

    private:
        //Doppler LUT for the refernce SLC
        isce3::core::LUT1d<double> _refDoppler;

        //Doppler LUT for the secondary SLC
        isce3::core::LUT1d<double> _secDoppler;

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
