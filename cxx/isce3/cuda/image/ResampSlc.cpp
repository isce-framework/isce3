#include "ResampSlc.h"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/LUT1d.h>

#include <isce3/image/Tile.h>

#include <isce3/cuda/core/gpuInterpolator.h>

#include "gpuResampSlc.h"

using isce3::io::Raster;

// Alternative generic resamp entry point: use filenames to internally create rasters
void isce3::cuda::image::ResampSlc::
resamp(const std::string & inputFilename,          // filename of input SLC
       const std::string & outputFilename,         // filename of output resampled SLC
       const std::string & rgOffsetFilename,       // filename of range offsets
       const std::string & azOffsetFilename,       // filename of azimuth offsets
       int inputBand, bool flatten, int rowBuffer,
       int chipSize)
{
    // Make input rasters
    Raster inputSlc(inputFilename, GA_ReadOnly);
    Raster rgOffsetRaster(rgOffsetFilename, GA_ReadOnly);
    Raster azOffsetRaster(azOffsetFilename, GA_ReadOnly);

    // Make output raster; geometry defined by offset rasters
    const size_t outLength = rgOffsetRaster.length();
    const size_t outWidth = rgOffsetRaster.width();
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32, "ISCE");

    // Call generic resamp
    resamp(inputSlc, outputSlc, rgOffsetRaster, azOffsetRaster, inputBand, flatten,
           rowBuffer, chipSize);
}

// Generic resamp entry point from externally created rasters
void isce3::cuda::image::ResampSlc::
resamp(isce3::io::Raster & inputSlc, isce3::io::Raster & outputSlc,
       isce3::io::Raster & rgOffsetRaster, isce3::io::Raster & azOffsetRaster,
       int inputBand, bool flatten, int rowBuffer,
       int chipSize)
{
    // Set the band number for input SLC
    _inputBand = inputBand;
    // Cache width of SLC image
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    // Cache output length and width from offset images
    const size_t outLength = rgOffsetRaster.length();
    const size_t outWidth = rgOffsetRaster.width();

    // Check if reference data is available
    if (flatten && !this->haveRefData()) {
        std::string error_msg{"Unable to flatten; reference data not provided."};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    // initialize interpolator
    isce3::cuda::core::gpuSinc2dInterpolator<thrust::complex<float>>
        interp(chipSize-1, isce3::core::SINC_SUB);

    // Determine number of tiles needed to process image
    const int nTiles = _computeNumberOfTiles(outLength, _linesPerTile);
    std::cout <<
        "GPU resampling using " << nTiles << " tiles of " << _linesPerTile
        << " lines per tile\n";
    // Start timer
    auto timerStart = std::chrono::steady_clock::now();

    // For each full tile of _linesPerTile lines...
    const isce3::core::LUT1d<double> dopplerLUT1d = isce3::core::avgLUT2dToLUT1d<double>(_dopplerLUT);
    for (int tileCount = 0; tileCount < nTiles; tileCount++)
    {
        // Make a tile for representing input SLC data
        Tile_t origSlcTile;
        origSlcTile.width(inWidth);
        // Set its line index bounds (line number in output image)
        origSlcTile.rowStart(tileCount * _linesPerTile);
        if (tileCount == (nTiles - 1)) {
            origSlcTile.rowEnd(outLength);
        } else {
            origSlcTile.rowEnd(origSlcTile.rowStart() + _linesPerTile);
        }

        // Initialize offsets tiles
        isce3::image::Tile<double> azOffTile, rgOffTile;
        _initializeOffsetTiles(origSlcTile, azOffsetRaster, rgOffsetRaster,
                               azOffTile, rgOffTile, outWidth);

        // Get corresponding image indices
        printf("Reading in image data for tile %d of %d\n", tileCount, nTiles);
        _initializeTile(origSlcTile, inputSlc, azOffTile, outLength, rowBuffer, chipSize/2);

        // Perform interpolation
        printf("Interpolating  tile %d of %d\n", tileCount, nTiles);
        gpuTransformTile(
                outputSlc,
                origSlcTile,
                rgOffTile,
                azOffTile,
                _rgCarrier,
                _azCarrier,
                dopplerLUT1d,
                interp,
                inWidth,
                inLength,
                this->startingRange(),
                this->rangePixelSpacing(),
                this->sensingStart(),
                this->prf(),
                this->wavelength(),
                this->refStartingRange(),
                this->refRangePixelSpacing(),
                this->refWavelength(),
                flatten,
                chipSize,
                _invalid_value);
    }

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    std::cout << "Elapsed processing time: " << elapsed << " sec" << "\n";
}
