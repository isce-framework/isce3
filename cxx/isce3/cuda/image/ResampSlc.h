#pragma once

#include "forward.h"

// isce3::image
#include <isce3/image/ResampSlc.h>

// Definition
class isce3::cuda::image::ResampSlc : public isce3::image::ResampSlc {

    public:
        // Meta-methods
        // Constructor from an isce3::product::Product
        inline ResampSlc(const isce3::product::Product &product, char frequency = 'A') :
            isce3::image::ResampSlc(product, frequency) {}

        // Constructor from an isce3::product::Product and reference product (flattening)
        inline ResampSlc(const isce3::product::Product & product,
                         const isce3::product::Product & refProduct,
                         char frequency = 'A') :
            isce3::image::ResampSlc(product, refProduct, frequency) {}

        /** Constructor from an isce3::product::RadarGridParameters (no flattening) */
        inline ResampSlc(const isce3::product::RadarGridParameters & rdr_grid,
                         const isce3::core::LUT2d<double> & doppler) :
            isce3::image::ResampSlc(rdr_grid, doppler) {}

        /** Constructor from an isce3::product::RadarGridParameters and reference radar grid (flattening) */
        inline ResampSlc(const isce3::product::RadarGridParameters & rdr_grid,
                         const isce3::product::RadarGridParameters & ref_rdr_grid,
                         const isce3::core::LUT2d<double> & doppler,
                         double wvl, double ref_wvl) :
            isce3::image::ResampSlc(rdr_grid, ref_rdr_grid, doppler, wvl, ref_wvl) {}

        // Constructor from individual components (no flattening)
        inline ResampSlc(const isce3::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl) :
            isce3::image::ResampSlc(doppler, startingRange, rangePixelSpacing, sensingStart,
                                   prf, wvl) {}

        // Constructor from individual components (flattening)
        inline ResampSlc(const isce3::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl,
                         double refStartingRange, double refRangePixelSpacing,
                         double refWvl) :
            isce3::image::ResampSlc(doppler, startingRange, rangePixelSpacing, sensingStart,
                                   prf, wvl, refStartingRange, refRangePixelSpacing, refWvl) {}

        // All resamp need? to be redefined to ensure derived functions used
        // Generic resamp entry point from externally created rasters
        void resamp(isce3::io::Raster & inputSlc, isce3::io::Raster & outputSlc,
                    isce3::io::Raster & rgOffsetRaster, isce3::io::Raster & azOffsetRaster,
                    int inputBand=1, bool flatten=false, bool isComplex=true, int rowBuffer=40,
                    int chipSize=isce3::core::SINC_ONE);

        // Generic resamp entry point: use filenames to create rasters
        void resamp(const std::string & inputFilename, const std::string & outputFilename,
                    const std::string & rgOffsetFilename, const std::string & azOffsetFilename,
                    int inputBand=1, bool flatten=false, bool isComplex=true, int rowBuffer=40,
                    int chipSize=isce3::core::SINC_ONE);

};
