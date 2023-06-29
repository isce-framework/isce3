#include "ResampSlc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <pyre/journal.h>

#include <isce3/core/Constants.h>

#include "Tile.h"

namespace isce3 { namespace image {

using isce3::io::Raster;

// Alternative generic resamp entry point: use filenames to internally create
// rasters
void ResampSlc::resamp(
        const std::string& inputFilename,    // filename of input SLC
        const std::string& outputFilename,   // filename of output resampled SLC
        const std::string& rgOffsetFilename, // filename of range offsets
        const std::string& azOffsetFilename, // filename of azimuth offsets
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
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32,
                     "ISCE");

    // Call generic resamp
    resamp(inputSlc, outputSlc, rgOffsetRaster, azOffsetRaster, inputBand,
           flatten, rowBuffer, chipSize);
}

// Generic resamp entry point from externally created rasters
void ResampSlc::resamp(isce3::io::Raster& inputSlc,
                       isce3::io::Raster& outputSlc,
                       isce3::io::Raster& rgOffsetRaster,
                       isce3::io::Raster& azOffsetRaster, int inputBand,
                       bool flatten, int rowBuffer,
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

    // Initialize resampling methods
    _prepareInterpMethods(isce3::core::SINC_METHOD, chipSize - 1);

    // Determine number of tiles needed to process image
    const size_t nTiles = _computeNumberOfTiles(outLength, _linesPerTile);
    std::cout << "Resampling using " << nTiles << " tiles of " << _linesPerTile
              << " lines per tile\n";
    // Start timer
    auto timerStart = std::chrono::steady_clock::now();

    // For each full tile of _linesPerTile lines...
    for (size_t tileCount = 0; tileCount < nTiles; tileCount++) {

        // Make a tile for representing input SLC data
        Tile_t tile;
        tile.width(inWidth);
        // Set its line index bounds (line number in output image)
        tile.rowStart(tileCount * _linesPerTile);
        if (tileCount == (nTiles - 1)) {
            tile.rowEnd(outLength);
        } else {
            tile.rowEnd(tile.rowStart() + _linesPerTile);
        }

        // Initialize offsets tiles
        Tile<float> azOffTile, rgOffTile;
        _initializeOffsetTiles(tile, azOffsetRaster, rgOffsetRaster, azOffTile,
                               rgOffTile, outWidth);

        // Get corresponding image indices
        std::cout << "Reading in image data for tile " << tileCount << "\n";
        _initializeTile(tile, inputSlc, azOffTile, outLength, rowBuffer,
                        chipSize / 2);

        // Perform interpolation
        std::cout << "Interpolating tile " << tileCount << "\n";
        _transformTile(tile, outputSlc, rgOffTile, azOffTile, inLength, flatten,
                       chipSize);
    }

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed =
            1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
                             timerEnd - timerStart)
                             .count();
    std::cout << "Elapsed processing time: " << elapsed << " sec\n";
}

// Initialize and read azimuth and range offsets
void ResampSlc::_initializeOffsetTiles(Tile_t& tile, Raster& azOffsetRaster,
                                       Raster& rgOffsetRaster,
                                       Tile<float>& azOffTile,
                                       Tile<float>& rgOffTile, size_t outWidth)
{
    // Copy size properties and initialize azimuth offset tiles
    azOffTile.width(outWidth);
    azOffTile.rowStart(tile.rowStart());
    azOffTile.rowEnd(tile.rowEnd());
    azOffTile.firstImageRow(tile.rowStart());
    azOffTile.lastImageRow(tile.rowEnd());
    azOffTile.allocate();

    // Do the same for range offset tiles
    rgOffTile.width(outWidth);
    rgOffTile.rowStart(tile.rowStart());
    rgOffTile.rowEnd(tile.rowEnd());
    rgOffTile.firstImageRow(tile.rowStart());
    rgOffTile.lastImageRow(tile.rowEnd());
    rgOffTile.allocate();

    // Read in block of range and azimuth offsets
    azOffsetRaster.getBlock(&azOffTile[0], 0, azOffTile.rowStart(),
                            azOffTile.width(), azOffTile.length());
    rgOffsetRaster.getBlock(&rgOffTile[0], 0, rgOffTile.rowStart(),
                            rgOffTile.width(), rgOffTile.length());
}

// Initialize tile bounds
void ResampSlc::_initializeTile(Tile_t& tile, Raster& inputSlc,
                                const Tile<float>& azOffTile, size_t outLength,
                                int rowBuffer, int chipHalf)
{
    // Cache geometry values
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    const size_t outWidth = azOffTile.width();

    // Compute minimum row index needed from input image
    tile.firstImageRow(outLength - 1);
    bool haveOffsets = false;
    for (size_t i = 0;
         i < std::min(static_cast<size_t>(rowBuffer), azOffTile.length());
         ++i) {
        for (size_t j = 0; j < outWidth; ++j) {
            // Get azimuth offset for pixel
            const double azOff = azOffTile(i, j);
            // Skip null values
            if (azOff < -5.0e5 || std::isnan(azOff)) {
                continue;
            } else {
                haveOffsets = true;
            }
            // Calculate corresponding minimum line index of input image.
            // Check for possible negative imageLine by checking if azOff < 0
            // and abs(azOff) + chipHalf > i + azOffTile.rowStart().
            // A negative value converted to a size_t incorrectly results in
            // std::numeric_limit<size_t>::max().
            // If a negative, then set imageLine to 0
            const bool isImageLineNegative =
                static_cast<size_t>(static_cast<int>(std::abs(azOff)) + chipHalf) >
                    i + azOffTile.rowStart() and azOff < 0;
            const size_t imageLine = isImageLineNegative ? 0 :
                static_cast<size_t>(i + azOff + azOffTile.rowStart() - chipHalf);
            // Update minimum row index
            tile.firstImageRow(std::min(tile.firstImageRow(), imageLine));
        }
    }
    // Final update
    if (haveOffsets) {
        tile.firstImageRow(std::max(tile.firstImageRow(), static_cast<size_t>(0)));
    } else {
        tile.firstImageRow(0);
    }

    // Compute maximum row index needed from input image
    tile.lastImageRow(0);
    haveOffsets = false;
    for (size_t i = std::max(azOffTile.length() - rowBuffer, static_cast<size_t>(0));
         i < azOffTile.length(); ++i) {
        for (size_t j = 0; j < outWidth; ++j) {
            // Get azimuth offset for pixel
            const double azOff = azOffTile(i, j);
            // Skip null values
            if (azOff < -5.0e5 || std::isnan(azOff)) {
                continue;
            } else {
                haveOffsets = true;
            }
            // Calculate corresponding minimum line index of input image
            const size_t imageLine = static_cast<size_t>(
                    i + azOff + azOffTile.rowStart() + chipHalf);
            // Update maximum row index
            tile.lastImageRow(std::max(tile.lastImageRow(), imageLine));
        }
    }
    // Final udpate
    if (haveOffsets) {
        tile.lastImageRow(std::min(tile.lastImageRow() + 1, inLength));
    } else {
        tile.lastImageRow(inLength);
    }

    // Tile will allocate memory for itself
    tile.allocate();

    // Read in tile.length() lines of data from the input image to the image
    // block
    inputSlc.getBlock(&tile[0], 0, tile.firstImageRow(), tile.width(),
                      tile.length(), _inputBand);

    // Remove carrier from input data
    for (size_t i = 0; i < tile.length(); i++) {
        const double az =  _sensingStart + (i + tile.firstImageRow()) / _prf;
        for (size_t j = 0; j < inWidth; j++) {
            const double rng = _startingRange + j * _rangePixelSpacing;
            // Evaluate the pixel's carrier phase
            const double phase = _rgCarrier.eval(az, rng)
                + _azCarrier.eval(az, rng);
            // Remove the carrier
            std::complex<float> cpxPhase(std::cos(phase), -std::sin(phase));
            tile(i, j) *= cpxPhase;
        }
    }
}

// Interpolate tile to perform transformation
void ResampSlc::_transformTile(Tile_t& tile, Raster& outputSlc,
                               const Tile<float>& rgOffTile,
                               const Tile<float>& azOffTile, size_t inLength,
                               bool flatten, int chipSize)
{
    if (flatten && !_haveRefData) {
        std::string error_msg{"Unable to flatten; reference data not provided."};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    // Cache geometry values
    const size_t inWidth = tile.width();
    const size_t outWidth = azOffTile.width();
    const size_t outLength = azOffTile.length();
    int chipHalf = chipSize / 2;

    // Allocate valarray for output image block
    std::valarray<std::complex<float>> imgOut(outLength * outWidth);
    // Initialize/fill with invalid values
    imgOut = _invalid_value;

    // From this point on, transformation is multithreaded
    size_t tileLine = 0;
    _Pragma("omp parallel shared(imgOut)")
    {

        // set half chip size
        // Allocate matrix for working sinc chip
        isce3::core::Matrix<std::complex<float>> chip(chipSize, chipSize);

        // Loop over lines to perform interpolation
        for (size_t i = tile.rowStart(); i < tile.rowEnd(); ++i) {

            // Compute azimuth time at i index
            const double az = _sensingStart + i / _prf;

            // Loop over width
            _Pragma("omp for") for (size_t j = 0; j < outWidth; ++j)
            {

                // Unpack offsets (units of bins)
                const float azOff = azOffTile(tileLine, j);
                const float rgOff = rgOffTile(tileLine, j);

                // Break into fractional and integer parts
                const size_t intAz = static_cast<size_t>(i + azOff);
                const size_t intRg = static_cast<size_t>(j + rgOff);
                const double fracAz = i + azOff - intAz;
                const double fracRg = j + rgOff - intRg;

                // Check bounds
                if ((intAz < chipHalf) || (intAz >= (inLength - chipHalf)))
                    continue;
                if ((intRg < chipHalf) || (intRg >= (inWidth - chipHalf)))
                    continue;

                // Slant range at j index
                const double rng = _startingRange + j * _rangePixelSpacing;

                // Check if the Doppler LUT covers the current position
                if (not _dopplerLUT.contains(az, rng))
                    continue;

                // Evaluate Doppler polynomial
                const double dop = _dopplerLUT.eval(az, rng) * 2 * M_PI / _prf;

                // Doppler to be added back. Simultaneously evaluate carrier
                // that needs to be added back after interpolation
                // Account for resample offsets in carrier evaluations.
                const double azPlusOffset = az
                    + static_cast<double>(azOff) / _prf;
                const double rngPlusOffset = rng
                    + static_cast<double>(rgOff) * _rangePixelSpacing;
                double phase = (dop * fracAz)
                    + _rgCarrier.eval(azPlusOffset, rngPlusOffset)
                    + _azCarrier.eval(azPlusOffset, rngPlusOffset);

                // Flatten the carrier phase if requested
                if (flatten && _haveRefData) {
                    phase += ((4. * (M_PI / _wavelength)) *
                              ((_startingRange - _refStartingRange) +
                               (j *
                                (_rangePixelSpacing - _refRangePixelSpacing)) +
                               (rgOff * _rangePixelSpacing))) +
                             ((4.0 * M_PI *
                               (_refStartingRange +
                                (j * _refRangePixelSpacing))) *
                              ((1.0 / _refWavelength) - (1.0 / _wavelength)));
                }

                // Read data chip without the carrier phases
                for (int ii = 0; ii < chipSize; ++ii) {
                    // Row to read from
                    const int chipRow =
                            intAz - tile.firstImageRow() + ii - chipHalf;
                    // Carrier phase
                    const double chipPhase = dop * (ii - chipHalf);
                    const std::complex<float> cval(
                            std::cos(chipPhase), -std::sin(chipPhase));
                    // Set the data values after removing doppler in azimuth
                    for (int jj = 0; jj < chipSize; ++jj) {
                        // Column to read from
                        const int chipCol = intRg + jj - chipHalf;
                        chip(ii, jj) = tile(chipRow, chipCol) * cval;
                    }
                }

                // Interpolate chip
                const std::complex<float> cval = _interp->interpolate(
                        chipHalf + fracRg, chipHalf + fracAz, chip);

                // Add doppler to interpolated value and save
                imgOut[tileLine * outWidth + j] =
                        cval *
                        std::complex<float>(std::cos(phase), std::sin(phase));

            } // end for over width

            // Update input line counter
            _Pragma("omp single") { ++tileLine; }

        } // end for over length

    } // end multithreaded block

    // Write block of data
    outputSlc.setBlock(imgOut, 0, tile.rowStart(), outWidth, outLength);
}

}} // namespace isce3::image
