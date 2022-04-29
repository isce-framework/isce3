#pragma once

#include "forward.h"

// isce3::image
#include <isce3/image/ResampSlc.h>

// Definition
class isce3::cuda::image::ResampSlc : public isce3::image::ResampSlc {

    public:
        // Meta-methods
        // Constructor from an isce3::product::RadarGridProduct
        inline ResampSlc(const isce3::product::RadarGridProduct &product, char frequency = 'A',
                         std::complex<float> invalid_value = std::complex<float>(0.0, 0.0)) :
            isce3::image::ResampSlc(product, frequency, invalid_value) {}

        // Constructor from an isce3::product::RadarGridProduct and reference product (flattening)
        inline ResampSlc(const isce3::product::RadarGridProduct & product,
                         const isce3::product::RadarGridProduct & refProduct,
                         char frequency = 'A',
                         std::complex<float> invalid_value = std::complex<float>(0.0, 0.0)) :
            isce3::image::ResampSlc(product, refProduct, frequency, invalid_value) {}

        /** Constructor from an isce3::product::RadarGridParameters (no flattening) */
        inline ResampSlc(const isce3::product::RadarGridParameters & rdr_grid,
                         const isce3::core::LUT2d<double> & doppler,
                         std::complex<float> invalid_value = std::complex<float>(0.0, 0.0)) :
            isce3::image::ResampSlc(rdr_grid, doppler, invalid_value) {}

        /** Constructor from an isce3::product::RadarGridParameters and reference radar grid (flattening) */
        inline ResampSlc(const isce3::product::RadarGridParameters & rdr_grid,
                         const isce3::product::RadarGridParameters & ref_rdr_grid,
                         const isce3::core::LUT2d<double> & doppler,
                         std::complex<float> invalid_value = std::complex<float>(0.0, 0.0)) :
            isce3::image::ResampSlc(rdr_grid, ref_rdr_grid, doppler, invalid_value) {}

        // Constructor from individual components (no flattening)
        inline ResampSlc(const isce3::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl,
                         std::complex<float> invalid_value = std::complex<float>(0.0, 0.0)) :
            isce3::image::ResampSlc(doppler, startingRange, rangePixelSpacing, sensingStart,
                                   prf, wvl, invalid_value) {}

        // Constructor from individual components (flattening)
        inline ResampSlc(const isce3::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl,
                         double refStartingRange, double refRangePixelSpacing,
                         double refWvl,
                         std::complex<float> invalid_value = std::complex<float>(0.0, 0.0)) :
            isce3::image::ResampSlc(doppler, startingRange, rangePixelSpacing, sensingStart,
                                   prf, wvl, refStartingRange, refRangePixelSpacing, refWvl,
                                   invalid_value) {}

        /* Generic resamp entry point from externally created rasters
         *
         * \param[in] inputSlc          raster of SLC to be resampled
         * \param[in] outputSlc         raster of resampled SLC
         * \param[in] rgOffsetRaster    raster of range shift to be applied
         * \param[in] azOffsetRaster    raster of azimuth shift to be applied
         * \param[in] inputBand         band of input raster to resample
         * \param[in] flatten           flag to flatten resampled SLC
         * \param[in] rowBuffer         number of rows excluded from top/bottom of azimuth
         *                              raster while searching for min/max indices of
         *                              resampled SLC
         * \param[in] chipSize          size of chip used in sinc interpolation
         */
        void resamp(isce3::io::Raster & inputSlc, isce3::io::Raster & outputSlc,
                    isce3::io::Raster & rgOffsetRaster, isce3::io::Raster & azOffsetRaster,
                    int inputBand=1, bool flatten=false, int rowBuffer=40,
                    int chipSize=isce3::core::SINC_ONE);

        /* Generic resamp entry point: use filenames to create rasters
         * internally in function.
         *
         * \param[in] inputFilename     path of file containing SLC to be resampled
         * \param[in] outputFilename    path of file containing resampled SLC
         * \param[in] rgOffsetFilename  path of file containing range shift to be applied
         * \param[in] azOffsetFilename  path of file containing azimuth shift to be applied
         * \param[in] inputBand         band of input raster to resample
         * \param[in] flatten           flag to flatten resampled SLC
         * \param[in] rowBuffer         number of rows excluded from top/bottom of azimuth
         *                              raster while searching for min/max indices of
         *                              resampled SLC
         * \param[in] chipSize          size of chip used in sinc interpolation
         */
        void resamp(const std::string & inputFilename, const std::string & outputFilename,
                    const std::string & rgOffsetFilename, const std::string & azOffsetFilename,
                    int inputBand=1, bool flatten=false, int rowBuffer=40,
                    int chipSize=isce3::core::SINC_ONE);

};
