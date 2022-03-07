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
        void Crossmul(const isce3::product::RadarGridProduct& referenceSLC,
                    const isce3::product::RadarGridProduct& secondarySLC,
                    const isce3::product::RadarGridProduct& outputInterferogram);
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

        /** Set dopplers LUT for reference SLC */
        inline void refDoppler(isce3::core::LUT1d<double> refDopp) { _refDoppler = refDopp; }

        /** Get doppler LUT for reference SLC */
        inline const isce3::core::LUT1d<double> & refDoppler() const { return _refDoppler; }

        /** Set dopplers LUT for secondary SLC */
        inline void secDoppler(isce3::core::LUT1d<double> secDopp) { _secDoppler = secDopp; }

        /** Get doppler LUT for secondary SLC */
        inline const isce3::core::LUT1d<double> & secDoppler() const { return _secDoppler; }

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double prf) { _prf = prf; }

        /** Get pulse repetition frequency (PRF) */
        inline double prf() const { return _prf; }

        /** Set range sampling frequency  */
        inline void rangeSamplingFrequency(double rgSamplingFreq) { _rangeSamplingFrequency = rgSamplingFreq; }

        /** Get range sampling frequency  */
        inline double rangeSamplingFrequency() const { return _rangeSamplingFrequency; }

        /** Set the range bandwidth */
        inline void rangeBandwidth(double rngBandwidth) { _rangeBandwidth = rngBandwidth; }

        /** Get the range bandwidth */
        inline double rangeBandwidth() const {return _rangeBandwidth; }

        /** Set range pixel spacing */
        inline void rangePixelSpacing(double rgPxlSpacing) { _rangePixelSpacing = rgPxlSpacing; }

        /** Get range pixel spacing */
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }

        /** Set Wavelength*/
        inline void wavelength(double wvl) { _wavelength = wvl; }

        /** Get Wavelength*/
        inline double wavelength() const { return _wavelength; }

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double azBandwidth) {_commonAzimuthBandwidth = azBandwidth; }

        /** Get azimuth common bandwidth */
        inline double commonAzimuthBandwidth() const { return _commonAzimuthBandwidth; }

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double beta) { _beta = beta; }

        /** Get beta parameter for the azimuth common band filter */
        inline double beta() const { return _beta; }

        /** Set number of range looks */
        inline void rangeLooks(int);

        /** Get number of range looks */
        inline int rangeLooks() const { return _rangeLooks; }

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const { return _azimuthLooks; }

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthBandFilter(bool doAzBandFilter) { _doCommonAzimuthBandFilter = doAzBandFilter; }

        /** Get common azimuth band filtering flag */
        inline bool doCommonAzimuthBandFilter() const { return _doCommonAzimuthBandFilter; }

        /** Set common range band filtering flag */
        inline void doCommonRangeBandFilter(bool doRgBandFilter) { _doCommonRangeBandFilter = doRgBandFilter; }

        /** Get common range band filtering flag */
        inline bool doCommonRangeBandFilter() const { return _doCommonRangeBandFilter; }

        /** Set oversample */
        inline void oversample(size_t oversamp) { _oversample = oversamp; }

        /** Get oversample */
        inline size_t oversample() const { return _oversample; }

        /** Set blockRows */
        inline void blockRows(size_t blockRows) { _blockRows = blockRows; }

        /** Get blockRows */
        inline size_t blockRows() const { return _blockRows; }

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
        size_t _blockRows = 8192;

        // upsampling factor
        size_t _oversample = 1;


};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC
