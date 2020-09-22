// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#pragma once

#include "forward.h"

#include <complex>
#include <isce3/core/LUT1d.h>
#include <isce3/io/forward.h>

/** \brief Intereferogram generation by cross-multiplication of reference and secondary SLCs.
 *
 *  The secondary SLC must be on the same image grid as the reference SLC, 
 */
class isce3::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};
        
        /*
        void Crossmul(const isce3::product::Product& referenceSLC,
                    const isce3::product::Product& secondarySLC,
                    const isce3::product::Product& outputInterferogram);
        */


        /** \brief Run crossmul */
        void crossmul(isce3::io::Raster& referenceSLC,
                    isce3::io::Raster& secondarySLC,
                    isce3::io::Raster& rngOffset,
                    isce3::io::Raster& interferogram,
                    isce3::io::Raster& coherence);

        /** \brief Run crossmul */
        void crossmul(isce3::io::Raster& referenceSLC, 
                    isce3::io::Raster& secondarySLC,
                    isce3::io::Raster& interferogram,
                    isce3::io::Raster& coherence);

        /** \brief Run crossmul */
        void crossmul(isce3::io::Raster& referenceSLC,
                    isce3::io::Raster& secondarySLC,
                    isce3::io::Raster& interferogram);

        /** Compute the frequency response due to a subpixel shift introduced by upsampling and downsampling*/
        void lookdownShiftImpact(size_t oversample, size_t fft_size,
                                size_t blockRows,
                                std::valarray<std::complex<float>> &shiftImpact);

        /** Range common band filtering*/
        void rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        std::valarray<std::complex<float>> geometryIfgram,
                        std::valarray<std::complex<float>> geometryIfgramConj,
                        std::valarray<std::complex<float>> &refSpectrum,
                        std::valarray<std::complex<float>> &secSpectrum,
                        std::valarray<double> &rangeFrequencies,
                        isce3::signal::Filter<float> &rngFilter,
                        size_t blockRows,
                        size_t ncols);
        

        /** Set doppler LUTs for reference and secondary SLCs*/
        inline void doppler(isce3::core::LUT1d<double>, 
                            isce3::core::LUT1d<double>);

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double);

        /** Get pulse repetition frequency (PRF) */
        inline double prf() const;

        /** Set range sampling frequency  */
        inline void rangeSamplingFrequency(double);

        /** Get range sampling frequency  */
        inline double rangeSamplingFrequency() const;

        /** Set the range bandwidth */
        inline void rangeBandwidth(double);

        /** Get the range bandwidth */
        inline double rangeBandwidth() const;

        /** Set range pixel spacing */
        inline void rangePixelSpacing(double);

        /** Get range pixel spacing */
        inline double rangePixelSpacing() const;

        /** Set Wavelength*/
        inline void wavelength(double);

        /** Get Wavelength*/
        inline double wavelength() const;

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double);

        /** Get azimuth common bandwidth */
        inline double commonAzimuthBandwidth() const;

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double);

        /** Get beta parameter for the azimuth common band filter */
        inline double beta() const;

        /** Set number of range looks */ 
        inline void rangeLooks(int);

        /** Get number of range looks */ 
        inline int rangeLooks() const;

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const;

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthBandFilter(bool);

        /** Get common azimuth band filtering flag */
        inline bool doCommonAzimuthBandFilter() const;

        /** Set common range band filtering flag */
        inline void doCommonRangeBandFilter(bool);

        /** Get common range band filtering flag */
        inline bool doCommonRangeBandFilter() const;

        /** Compute the avergae frequency shift in range direction between two SLCs*/
        inline void rangeFrequencyShift(std::valarray<std::complex<float>> &refAvgSpectrum,
                std::valarray<std::complex<float>> &secAvgSpectrum,
                std::valarray<double> &rangeFrequencies,
                size_t blockRowsData,
                size_t fft_size,
                double &frequencyShift);

        /** estimate the index of the maximum of a vector of data */
        inline void getPeakIndex(std::valarray<float> data, 
                                size_t &peakIndex);

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

        // Flag for computing coherence
        bool _computeCoherence = true;

        // number of lines per block
        size_t blockRows = 8192;

        // upsampling factor
        size_t oversample = 1;

        
};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC
