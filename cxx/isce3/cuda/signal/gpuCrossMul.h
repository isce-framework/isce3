#pragma once

#include "forward.h"
#include <isce3/io/forward.h> // Raster

#include <isce3/core/Common.h>
#include <isce3/core/LUT1d.h>
#include <thrust/complex.h>

namespace isce3::cuda::signal {

class gpuCrossmul {

    public:
        gpuCrossmul() {};
        ~gpuCrossmul() {};

        /**
         * Crossmultiply 2 SLCs
         *
         * \param[in]  refSlcRaster input raster of reference SLC
         * \param[in]  secSlcRaster input raster of secondary SLC
         * \param[out] ifgRaster    output interferogram raster
         * \param[out]  coherenceRaster output coherence raster
         * \param[in]  rngOffsetRaster  optional pointer to range offset raster
         *                              if provided, interferogram will be flattened
         */
        void crossmul(isce3::io::Raster& refSlcRaster,
                isce3::io::Raster& secSlcRaster,
                isce3::io::Raster& ifgRaster,
                isce3::io::Raster& coherenceRaster,
                isce3::io::Raster* rngOffsetRaster = nullptr) const;

        /** Set doppler LUTs for reference and secondary SLCs*/
        void doppler(isce3::core::LUT1d<double> refDoppler,
                isce3::core::LUT1d<double> secDoppler);

        /** Set reference doppler */
        inline void refDoppler(isce3::core::LUT1d<double> refDopp) {_refDoppler = refDopp;};

        /** Get reference doppler */
        inline const isce3::core::LUT1d<double> & refDoppler() const {return _refDoppler;};

        /** Set secondary doppler */
        inline void secDoppler(isce3::core::LUT1d<double> secDopp) {_secDoppler = secDopp;};

        /** Get secondary doppler */
        inline const isce3::core::LUT1d<double> & secDoppler() const {return _secDoppler;};

        /** Set reference and secondary starting range shift */
        inline void startingRangeShift(double rng_shift) { _offsetStartingRangeShift = rng_shift; }

        /** Get reference and secodnary starting range shift */
        inline double startingRangeShift() const { return _offsetStartingRangeShift; }

        /** Set range pixel spacing */
        inline void rangePixelSpacing(double rngPxl) {_rangePixelSpacing = rngPxl;};

        /** Get range pixel spacing */
        inline double rangePixelSpacing() const {return _rangePixelSpacing;};

        /** Set Wavelength*/
        inline void wavelength(double v) {_wavelength = v;};

        /** Get Wavelength*/
        inline double wavelength() const {return _wavelength;};

        /** Set number of range looks */
        void rangeLooks(int rngLks);

        /** Get number of range looks */
        inline int rangeLooks() const {return _rangeLooks;};

        /** Set number of azimuth looks */
        void azimuthLooks(int azLks);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const {return _azimuthLooks;};

        /** Set oversample factor */
        inline void oversampleFactor(size_t v) {_oversampleFactor = v;};

        /** Get oversample factor */
        inline size_t oversampleFactor() const {return _oversampleFactor;};

        /** Set linesPerBlock*/
        inline void linesPerBlock(size_t v) {_linesPerBlock = v;};

        /** Get linesPerBlock*/
        inline size_t linesPerBlock() const {return _linesPerBlock;};

        /** Get multilook flag */
        inline bool multiLookEnabled() const { return _multiLookEnabled; }

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

} // namespace isce3::cuda::signal
