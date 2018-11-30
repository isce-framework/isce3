// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#ifndef ISCE_LIB_CROSSMUL_H
#define ISCE_LIB_CROSSMUL_H

# include <assert.h>

// pyre
#include <portinfo>
#include <pyre/journal.h>

#include <isce/io/Raster.h>
#include <isce/core/Poly2d.h>
#include "Signal.h"
#include "Filter.h"


namespace isce {
    namespace signal {
        class Crossmul;
    }
}

/** \brief Intereferogram generation by cross-multiplication of reference and secondary SLCs.
 *
 *  The secondary SLC must be on the same image grid as the reference SLC, 
 */
class isce::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};
        
        /*
        void Crossmul(const isce::product::Product& referenceSLC,
                    const isce::product::Product& secondarySLC,
                    const isce::product::Product& outputInterferogram);
        */


        /** \brief Run crossmul */
        void crossmul(isce::io::Raster& referenceSLC,
                    isce::io::Raster& secondarySLC,
                    isce::io::Raster& interferogram,
		    isce::io::Raster& rngOffset);
        
        void crossmul(isce::io::Raster& referenceSLC, 
                    isce::io::Raster& secondarySLC,
                    isce::io::Raster& interferogram);

        /** */
        void lookdownShiftImpact(size_t oversample, size_t nfft, 
                                size_t blockRows,
                                std::valarray<std::complex<float>> &shiftImpact);

        void rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        std::valarray<std::complex<float>> geometryIfgram,
                        std::valarray<std::complex<float>> geometryIfgramConj,
                        std::valarray<std::complex<float>> &refSpectrum,
                        std::valarray<std::complex<float>> &secSpectrum,
                        std::valarray<double> &rangeFrequencies,
                        isce::signal::Filter<float> &rngFilter,
                        size_t blockRows,
                        size_t ncols);
        

       /** Set doppler polynomials for reference and secondary SLCs*/
        inline void doppler(isce::core::Poly2d, 
                            isce::core::Poly2d);

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double);

        /** Set range sampling frequency  */
        inline void rangeSamplingFrequency(double);

        /** Set the range bandwidth */
        inline void rangeBandwidth(double);

        /** Range pixel spacing */
        inline void rangePixelSpacing(double);

        /** Set Wavelength*/
        inline void wavelength(double);


        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double);

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double);


        /** Set number of range looks */ 
        inline void rangeLooks(int);

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthbandFiltering(bool);

        /** Set common range band filtering flag */
        inline void doCommonRangebandFiltering(bool);

        inline void rangeFrequencyShift(std::valarray<std::complex<float>> &refAvgSpectrum,
                                        std::valarray<std::complex<float>> &secAvgSpectrum,
                                        std::valarray<double> &rangeFrequencies,
                                        size_t blockRowsData,
                                        size_t nfft,
                                        double &frequencyShift);

        inline void getPeakIndex(std::valarray<float> data, 
                                size_t &peakIndex);

    private:
        //Doppler polynomial for the refernce SLC
        isce::core::Poly2d _refDoppler;

        //Doppler polynomial for the secondary SLC
        isce::core::Poly2d _secDoppler;

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
        int _rangeLooks;

        // number of azimuth looks
        int _azimuthLooks;

        // Flag for common azimuth band filtering
        bool _doCommonAzimuthbandFilter;

        // Flag for common range band filtering
        bool _doCommonRangebandFilter;

        // number of lines per block
        size_t blockRows = 1000;

        // upsampling factor
        size_t oversample = 2;
};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC

#endif
