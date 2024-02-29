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

        /**
         * Crossmultiply 2 SLCs
         *
         * \param[in]  refSlcRaster input raster of reference SLC
         * \param[in]  secSlcRaster input raster of secondary SLC
         * \param[out] ifgRaster    output interferogram raster
         * \param[out] coherenceRaster  output coherence raster
         * \param[in]  rngOffsetRaster  optional pointer to range offset raster
         *                              if provided, interferogram will be flattened
         */
        void crossmul(isce3::io::Raster& refSlcRaster,
                    isce3::io::Raster& secSlcRaster,
                    isce3::io::Raster& ifgRaster,
                    isce3::io::Raster& coherence,
                    isce3::io::Raster* rngOffsetRaster = nullptr) const;

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

        /** Set reference and seconary starting range shift */
        inline void startingRangeShift(double rng_shift) { _offsetStartingRangeShift = rng_shift; }

        /** Get reference and secondary starting range shift */
        inline double startingRangeShift() const { return _offsetStartingRangeShift; }

        /** Set range pixel spacing */
        inline void rangePixelSpacing(double rgPxlSpacing) { _rangePixelSpacing = rgPxlSpacing; }

        /** Get range pixel spacing */
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }

        /** Set Wavelength*/
        inline void wavelength(double wvl) { _wavelength = wvl; }

        /** Get Wavelength*/
        inline double wavelength() const { return _wavelength; }

        /** Set number of range looks */
        inline void rangeLooks(int);

        /** Get number of range looks */
        inline int rangeLooks() const { return _rangeLooks; }

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const { return _azimuthLooks; }

        /** Set oversample factor */
        inline void oversampleFactor(size_t oversamp) { _oversampleFactor = oversamp; }

        /** Get oversample factor */
        inline size_t oversampleFactor() const { return _oversampleFactor; }

        /** Set linesPerBlock */
        inline void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

        /** Get linesPerBlock */
        inline size_t linesPerBlock() const { return _linesPerBlock; }

        /** Get boolean multilook flag */
        inline bool multiLookEnabled() const { return _multiLookEnabled; }

        /** Compute the avergae frequency shift in range direction between two SLCs*/
        inline void rangeFrequencyShift(std::valarray<std::complex<float>> &refAvgSpectrum,
                std::valarray<std::complex<float>> &secAvgSpectrum,
                std::valarray<double> &rangeFrequencies,
                size_t linesPerBlockData,
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

        // starting range shifts between the secondary and reference RSLC in meters
        double _offsetStartingRangeShift = 0.0;

        // range pixel spacing
        double _rangePixelSpacing;

        // radar wavelength
        double _wavelength;

        // number of range looks
        int _rangeLooks = 1;

        // number of azimuth looks
        int _azimuthLooks = 1;

        bool _multiLookEnabled = false;

        // number of lines per block
        size_t _linesPerBlock = 1024;

        // upsampling factor
        size_t _oversampleFactor = 1;


};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC
